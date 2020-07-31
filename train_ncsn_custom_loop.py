import numpy as np
import tensorflow as tf
from ncsn import score_network
from flow_models import utils
from pipeline import data_loader
import tensorflow_datasets as tfds
import argparse
import time
import datetime
import os
import sys
from train_utils import *
tfk = tf.keras


def get_uncompiled_model(args):
    # inputs
    perturbed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network.CondRefineNetDilated(args.data_shape, args.n_filters,
                                                 args.num_classes, args.use_logit)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name="ScoreNetwork")

    return model


def anneal_langevin_dynamics(x_mod, data_shape, model, n_samples, sigmas, n_steps_each=100, step_lr=2e-5, return_arr=False):
    """
    Anneal Langevin dynamics
    """
    if return_arr:
        x_arr = tf.expand_dims(x_mod, axis=0).numpy()
    for i, sigma in enumerate(sigmas):
        labels = tf.ones(n_samples, dtype=tf.int32) * i
        step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
        for s in range(n_steps_each):
            noise = tf.random.normal([n_samples] + list(data_shape)) * tf.math.sqrt(step_size * 2)
            grad = model([x_mod, labels], training=True)
            x_mod = x_mod + step_size * grad + noise
            if return_arr:
                if ((s + 1) % (n_steps_each // 10) == 0):
                    x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()


def train(model, optimizer, sigmas_np, mirrored_strategy, distr_train_dataset, distr_eval_dataset,
          train_summary_writer, test_summary_writer, args):
    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)
    with mirrored_strategy.scope():
        def compute_train_loss(scores, target, sample_weight):
            per_example_loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

        def compute_test_loss(scores, target, sample_weight):
            per_example_loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

    def get_noise_conditionned_data(X):
        sigma_idx = tf.random.uniform(shape=(args.batch_size,), maxval=args.num_classes, dtype=tf.int32)
        used_sigma = tf.gather(params=sigmas_tf, indices=sigma_idx)
        used_sigma = tf.reshape(used_sigma, (args.batch_size, 1, 1, 1))
        perturbed_X = X + tf.random.normal([args.batch_size] + args.data_shape) * used_sigma
        inputs = {'perturbed_X': perturbed_X, 'sigma_idx': sigma_idx}
        target = -(perturbed_X - X) / (used_sigma ** 2)
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
    loss_per_epoch = 10
    count_step = optimizer.iterations.numpy()
    for epoch in range(args.n_epochs):
        epoch_loss_avg.reset_states()

        if is_nan_loss:
            break

        for batch in distr_train_dataset:

            loss = distributed_train_step(batch)
            history_loss_avg.update_state(loss)
            epoch_loss_avg.update_state(loss)
            count_step += 1

            # every loss_per_epoch train step
            if count_step % (args.n_train // (args.batch_size * loss_per_epoch)) == 0:
                # check nan loss
                if (tf.math.is_nan(loss)) or (tf.math.is_inf(loss)):
                    print('Nan or Inf Loss: {}'.format(loss))
                    is_nan_loss = True
                    break

                # Save history and monitor it on tensorboard
                curr_loss_history = history_loss_avg.result()
                loss_history.append(curr_loss_history)
                with train_summary_writer.as_default():
                    step_int = int(loss_per_epoch * count_step * batch_size / n_train)
                    tf.summary.scalar(
                        'loss', curr_loss_history, step=step_int)

                history_loss_avg.reset_states()

            # 10 times during training
            if (args.n_epochs < 10) or (epoch % (args.n_epochs // 10)) == 0:
                test_loss.reset_states()

                # Compute validation loss
                for elt in distr_eval_dataset:
                    test_loss.update_state(distributed_test_step(elt))

                step_int = int(loss_per_epoch * count_step * args.batch_size / args.n_train)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=step_int)
                print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                    epoch, epoch_loss_avg.result(), test_loss.result()))

                # Generate some samples and visualize them on tensorboard
                x_mod = tf.random.uniform([32] + args.data_shape)
                if args.use_logit:
                    x_mod = (1. - 2. * args.alpha) * x_mod + args.alpha
                    x_mod = tf.math.log(x_mod) - tf.math.log(1. - x_mod)
                if mirrored_strategy is not None:
                    with mirrored_strategy.scope():
                        gen_samples = anneal_langevin_dynamics(x_mod, args.data_shape, model,
                                                               32, sigmas_np, return_arr=True)

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

def main(args):

    sigmas_np = np.logspace(np.log(args.sigma1) / np.log(10), np.log(args.sigmaL) / np.log(10), num=args.num_classes)
    # sigmas_np = np.linspace(args.sigma1, args.sigmaL, num=args.num_classes)

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

        output_dirname = 'ncsn' + '_' + dataset + '_' + \
            str(args.n_filters) + '_' + str(args.batch_size)

        if args.use_logit:
            output_dirname += '_logit'
        if args.data_type == 'melspec':
            output_dirname += '_' + args.scale
        if args.restore is not None:
            output_dirname += '_ctd'

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
                                                                            reshuffle=True, batch_size=None, mirrored_strategy=None)
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
        if args.data_type == 'image':
            X = tf.cast(X['image'], tf.float32)
            if args.dataset == 'mnist':
                X = tf.pad(X, tf.constant([[2, 2], [2, 2], [0, 0]]))
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

    # Build ScoreNet
    with mirrored_strategy.scope():
        model = get_uncompiled_model(args)

    # Set up checkpoint
    ckpt, manager, manager_issues = setUp_checkpoint(
        mirrored_strategy, args, model, optimizer)

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
    total_trainable_variables = utils.total_trainable_variables(model)
    print("Total Trainable Variables: ", total_trainable_variables)
    train(model, optimizer, sigmas_np, mirrored_strategy, distr_train_dataset, distr_eval_dataset,
          train_summary_writer, test_summary_writer, args)
    print("Training time: ", np.round(time.time() - t0, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NCSN model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")
    parser.add_argument("--use_logit", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-6)

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=str, default="power", help="power or dB")

    # Output and Restore Directory
    parser.add_argument('--output', type=str, default='trained_ncsn',
                        help='output dirpath for savings')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--debug', action="store_true")

    # Model hyperparameters
    parser.add_argument("--n_filters", type=int, default=64,
                        help='number of filters in the Score Network')
    parser.add_argument("--sigma1", type=float, default=1.)
    parser.add_argument("--sigmaL", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=10)

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    args = parser.parse_args()

    main(args)
