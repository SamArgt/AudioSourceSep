import numpy as np
import tensorflow as tf
from ncsn.utils import *
from flow_models.utils import *
from datasets import data_loader
import tensorflow_datasets as tfds
import argparse
import time
import os
import sys
from train_utils import *
import tensorflow_addons as tfa
tfk = tf.keras


"""
Script for training a NCSN model on MNIST, CIFAR-10 or custom MelSpectrograms Dataset
"""


def train(model, optimizer, sigmas_np, mirrored_strategy, distr_train_dataset, distr_eval_dataset,
          train_summary_writer, test_summary_writer, manager, args):
    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)

    with mirrored_strategy.scope():
        def compute_train_loss(scores, target, sample_weight):
            per_example_loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
            per_example_loss *= sample_weight
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

        def compute_test_loss(scores, target, sample_weight):
            per_example_loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
            per_example_loss *= sample_weight
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

    def get_noise_conditionned_data(X):
        local_batch_size = X.shape[-1]
        sigma_idx = tf.random.uniform(shape=(local_batch_size,), maxval=args.num_classes, dtype=tf.int32)
        used_sigma = tf.gather(params=sigmas_tf, indices=sigma_idx)
        used_sigma = tf.reshape(used_sigma, (local_batch_size, 1, 1, 1))
        noise = tf.random.normal([local_batch_size] + args.data_shape) * used_sigma
        perturbed_X = X + noise
        inputs = {'perturbed_X': perturbed_X, 'sigma_idx': sigma_idx}
        target = - 1 / (used_sigma ** 2) * noise
        sample_weight = used_sigma ** 2
        return inputs, target, sample_weight

    def train_step(X):
        inputs, target, sample_weight = get_noise_conditionned_data(X)
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            scores = model(inputs)
            loss = compute_train_loss(scores, target, sample_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            list(zip(gradients, model.trainable_variables)))
        return loss

    def test_step(X):
        inputs, target, sample_weight = get_noise_conditionned_data(X)
        scores = model(inputs)
        loss = compute_test_loss(scores, target, sample_weight)
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            train_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            test_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def post_processing(x):
        if args.use_logit:
            x = 1. / (1. + np.exp(-x))
            x = (x - args.alpha) / (1. - 2. * args.alpha)
        x = x * (args.maxval - args.minval) + args.minval
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
            if args.scale == "power":
                x = librosa.power_to_db(x)
        return x

    test_loss = tfk.metrics.Mean(name='test_loss')
    history_loss_avg = tf.keras.metrics.Mean(name="tensorboard_loss")
    epoch_loss_avg = tf.keras.metrics.Mean(name="epoch_loss")
    is_nan_loss = False
    loss_per_epoch = 5
    count_step = optimizer.iterations.numpy()
    min_test_loss = 1e6
    for epoch in range(args.n_epochs):
        epoch_loss_avg.reset_states()

        if is_nan_loss:
            break

        for batch in distr_train_dataset:

            loss = distributed_train_step(batch)
            history_loss_avg.update_state(loss)
            epoch_loss_avg.update_state(loss)

            count_step += 1

            # every loss_per_epoch iterations
            if count_step % (args.n_train // (args.batch_size * loss_per_epoch)) == 0:
                # check nan loss
                if (tf.math.is_nan(loss)) or (tf.math.is_inf(loss)):
                    print('Nan or Inf Loss: {}'.format(loss))
                    is_nan_loss = True
                    break

                # Save history and monitor it on tensorboard
                curr_loss_history = history_loss_avg.result()
                with train_summary_writer.as_default():
                    step_int = int(10 * count_step * args.batch_size / args.n_train)
                    tf.summary.scalar(
                        'loss', curr_loss_history, step=step_int)

                history_loss_avg.reset_states()

        # Every 10 epochs: Compute validation loss
        if (epoch % 10) == 0:
            test_loss.reset_states()

            for elt in distr_eval_dataset:
                test_loss.update_state(distributed_test_step(elt))

            step_int = int(10 * count_step * args.batch_size / args.n_train)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=step_int)
            print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                epoch, epoch_loss_avg.result(), test_loss.result()))

            # If minimum validation loss: save model
            cur_test_loss = test_loss.result()
            if cur_test_loss < min_test_loss:
                min_test_loss = cur_test_loss
                save_path = manager.save()
                print("Model Saved at {}".format(save_path))

        # Every 50 epochs: Generate Samples
        if ((epoch % 50) == 0 and epoch > 0) or (epoch == args.n_epochs - 1):
            x_mod = tf.random.uniform([32] + args.data_shape)
            if args.use_logit:
                x_mod = (1. - 2. * args.alpha) * x_mod + args.alpha
                x_mod = tf.math.log(x_mod) - tf.math.log(1. - x_mod)

            if mirrored_strategy is not None:
                with mirrored_strategy.scope():
                    gen_samples = anneal_langevin_dynamics(x_mod, args.data_shape, model,
                                                           32, sigmas_np, n_steps_each=args.T, step_lr=args.step_lr,
                                                           return_arr=True)

            gen_samples = post_processing(gen_samples)
            np.save(os.path.join("generated_samples", "generated_samples_{}".format(epoch)), gen_samples)
            try:
                figure = image_grid(gen_samples[-1, :, :, :], args.data_shape, args.data_type,
                                    sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
                with train_summary_writer.as_default():
                    tf.summary.image("32 generated samples", plot_to_image(figure),
                                     max_outputs=50, step=epoch)
            except IndexError:
                with train_summary_writer.as_default():
                    tf.summary.text(name="display error",
                                    data="Impossible to display spectrograms because of NaN values",
                                    step=epoch)

            save_path = manager.save()
            print("Model Saved at {}".format(save_path))

    save_path = manager.save()
    print("Model Saved at {}".format(save_path))

def main(args):

    if args.config is not None:
        new_args = get_config(args.config)
        new_args.dataset = args.dataset
        new_args.output = args.output
        new_args.debug = args.debug
        new_args.restore = args.restore
        args = new_args

    sigmas_np = get_sigmas(args.sigma1, args.sigmaL, args.num_classes)
    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)

    # miscellaneous paramaters
    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.data_type = "image"
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.imgdata_type_type = "image"
    else:
        args.data_shape = [args.height, args.width, 1]
        args.dataset = os.path.abspath(args.dataset)
        args.data_type = "melspec"
        args.instrument = os.path.split(args.dataset)[-1]

    # output directory
    if args.restore is not None:
        abs_restore_path = os.path.abspath(args.restore)
    if args.output == 'trained_ncsn':
        if args.dataset != 'mnist' and args.dataset != 'cifar10':
            dataset = args.instrument
        else:
            dataset = args.dataset

        output_dirname = 'ncsn' + args.version + '_' + dataset + '_' + \
            str(args.n_filters) + '_' + str(args.batch_size)

        if args.use_logit:
            output_dirname += '_logit'
        if args.data_type == 'melspec':
            output_dirname += '_' + args.scale
        if args.restore is not None:
            output_dirname += '_ctd'

        # output_dirname += '_custom_loop'

        output_dirpath = os.path.join(args.output, output_dirname)
    else:
        output_dirpath = args.output
    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)
    try:
        os.mkdir("generated_samples")
    except FileExistsError:
        pass
    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))
    args.local_batch_size = args.batch_size // mirrored_strategy.num_replicas_in_sync

    # Load and Preprocess Dataset
    if (args.dataset == "mnist") or (args.dataset == "cifar10"):
        datasets = tfds.load(name=args.dataset, with_info=False, as_supervised=False)
        ds_train, ds_test = datasets['train'], datasets['test']
        if args.dataset == "mnist":
            args.n_train = 60000
        else:
            args.n_train = 50000
        args.n_test = 10000
        args.minval = 0.
        args.maxval = 256.
        args.sampling_rate, args.fmin, args.fmax = None, None, None

    else:
        ds_train, ds_test, _, n_train, n_test = data_loader.load_melspec_ds(args.dataset + '/train', args.dataset + '/test',
                                                                            shuffle=True, batch_size=None, mirrored_strategy=None)
        args.fmin = 125
        args.fmax = 7600
        args.sampling_rate = 16000
        if args.scale == 'power':
            args.maxval = 100.
            args.minval = 1e-10
        elif args.scale == 'dB':
            args.maxval = 20.
            args.minval = -100.
        else:
            raise ValueError("scale should be 'power' or 'dB'")
        args.n_train = n_train
        args.n_test = n_test

    BUFFER_SIZE = 10000
    BATCH_SIZE = args.batch_size

    def map_fn(X):
        X = (X - args.minval) / (args.maxval - args.minval)
        if args.use_logit:
            X = X * (1. - 2 * args.alpha) + args.alpha
            X = tf.math.log(X) - tf.math.log(1. - X)
        return X

    train_dataset = ds_train.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    distr_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    eval_dataset = ds_test.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    eval_dataset = eval_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = eval_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    distr_eval_dataset = mirrored_strategy.experimental_distribute_dataset(eval_dataset)

    # Set up tensorboard
    train_summary_writer, test_summary_writer = setUp_tensorboard()

    # Display original images
    def post_processing(x):
        if args.use_logit:
            x = 1. / (1. + np.exp(-x))
            x = (x - args.alpha) / (1. - 2. * args.alpha)
        x = x * (args.maxval - args.minval) + args.minval
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
            if args.scale == "power":
                x = librosa.power_to_db(x)
        return x
    with train_summary_writer.as_default():
        sample = post_processing(list(train_dataset.take(1).as_numpy_iterator())[0])
        figure = image_grid(sample, args.data_shape, args.data_type,
                            sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
        tf.summary.image("original images", plot_to_image(figure), max_outputs=1, step=0)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)
    if args.ema:
        optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.999)

    # Build ScoreNet
    with mirrored_strategy.scope():
        if args.version == 'v2':
            model = get_uncompiled_model_v2(args, sigmas=sigmas_tf)
        elif args.version == 'v1':
            model = get_uncompiled_model(args)
        else:
            raise ValueError('version should be "v1" or "v2"')

    # Set up checkpoint
    ckpt, manager = setUp_checkpoint(mirrored_strategy, model, optimizer, max_to_keep=10)

    # restore
    if args.restore is not None:
        with mirrored_strategy.scope():
            if args.latest:
                checkpoint_restore_path = tf.train.latest_checkpoint(abs_restore_path)
                assert checkpoint_restore_path is not None, abs_restore_path
            else:
                checkpoint_restore_path = abs_restore_path
            status = ckpt.restore(checkpoint_restore_path)
            status.assert_existing_objects_matched()
            assert optimizer.iterations > 0
            print("Model Restored from {}".format(abs_restore_path))

    # Display parameters
    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with train_summary_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)

    # Train
    tot_trainable_variables = total_trainable_variables(model)
    print("Total Trainable Variables: ", tot_trainable_variables)
    t0 = time.time()
    train(model, optimizer, sigmas_np, mirrored_strategy, distr_train_dataset, distr_eval_dataset,
          train_summary_writer, test_summary_writer, manager, args)
    print("Training time: ", np.round(time.time() - t0, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NCSN model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")

    # Output and Restore Directory
    parser.add_argument('--output', type=str, default='trained_ncsn',
                        help='output dirpath for savings')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--debug', action="store_true")

    # config
    parser.add_argument('--config', type=str, help='path to the config file. Overwrite all other parameters below')

    parser.add_argument("--use_logit", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-6)

    # NCSN V2
    parser.add_argument('--version', type=str, help='Version of NCSN', default='v2')
    parser.add_argument('--ema', action='store_true', help="Use Exponential Moving Average")

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=str, default="dB", help="power or dB")

    # Model hyperparameters
    parser.add_argument("--n_filters", type=int, default=192,
                        help='number of filters in the Score Network')
    parser.add_argument("--sigma1", type=float, default=55.)
    parser.add_argument("--sigmaL", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=325)

    # Langevin Dynamics parameters
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--step_lr', type=float, default=5.5e-6)

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
