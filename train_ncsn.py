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


def anneal_langevin_dynamics(data_shape, model, n_samples, sigmas, n_steps_each=100, step_lr=2e-5, return_arr=False):
    """
    Anneal Langevin dynamics
    """
    x_mod = tf.random.uniform([n_samples] + list(data_shape))
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
                x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()


class CustomLoss(tfk.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def __call__(self, scores, target, sample_weight=None):
        loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
        if sample_weight is not None:
            return tf.reduce_mean(loss * sample_weight)
        else:
            return tf.reduce_mean(loss)


def main(args):

    sigmas_np = np.logspace(np.log(args.sigma1) / np.log(10),
                            np.log(args.sigmaL) / np.log(10),
                            num=args.num_classes)

    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)

    # miscellaneous paramaters
    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.img_type = "image"
        args.preprocessing_glow = None
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.img_type = "image"
        args.preprocessing_glow = None
    else:
        args.data_shape = [args.height, args.width, 1]
        args.dataset = os.path.abspath(args.dataset)
        args.img_type = "melspec"
        args.preprocessing_glow = "melspec"
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
    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Distributed Strategy
    if args.mirrored_strategy:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(
            mirrored_strategy.num_replicas_in_sync))
        args.local_batch_size = args.batch_size // mirrored_strategy.num_replicas_in_sync
    else:
        mirrored_strategy = None
        args.local_batch_size = args.batch_size

    # Load and Preprocess Dataset
    if (args.dataset == "mnist") or (args.dataset == "cifar10"):
        datasets = tfds.load(name=args.dataset, with_info=False, as_supervised=False)
        ds_train, ds_test = datasets['train'], datasets['test']
        args.dataset_maxval = 256.

    else:
        ds_train, ds_test, _, n_train, n_test = data_loader.load_melspec_ds(args.dataset + '/train', args.dataset + '/test',
                                                                            reshuffle=True, batch_size=None, mirrored_strategy=None)
        args.dataset_maxval = 100.1
        args.n_train = n_train
        args.n_test = n_test

    BUFFER_SIZE = 10000
    BATCH_SIZE = args.batch_size

    def preprocess(x):
        sigma_idx = tf.random.uniform(shape=(), maxval=args.num_classes, dtype=tf.int32)
        used_sigma = tf.gather(params=sigmas_tf, indices=sigma_idx)
        X = tf.cast(x['image'], tf.float32)
        if args.dataset == 'mnist':
            X = tf.pad(X, tf.constant([[2, 2], [2, 2], [0, 0]]))
        if args.img_type == 'image':
            X = (X + tf.random.uniform(args.data_shape)) / args.dataset_maxval
        else:
            X /= args.dataset_maxval
        if args.use_logit:
            X = X * (1. - 2 * args.alpha) + args.alpha
            X = tf.math.log(X) - tf.math.log(1. - X)
        perturbed_X = X + tf.random.normal(args.data_shape) * used_sigma
        inputs = {'perturbed_X': perturbed_X, 'sigma_idx': sigma_idx}
        target = -(perturbed_X - X) / (used_sigma ** 2)
        sample_weight = used_sigma ** 2
        return inputs, target, sample_weight

    train_dataset = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    eval_dataset = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_dataset = eval_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = eval_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Display original images
    # Clear any logs from previous runs
    try:
        shutil.rmtree('logs')
    except FileNotFoundError:
        pass
    logdir = os.path.join("logs", "samples") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        inputs, _, _ = list(train_dataset.take(1).as_numpy_iterator())[0]
        sample = inputs["perturbed_X"]
        if args.use_logit:
            sample = 1. / (1. + np.exp(-sample))
            sample = (sample - args.alpha) / (1. - args.alpha)
        if args.img_type == 'melspec':
            sample *= args.dataset_maxval
        sample = sample[:32]
        figure = image_grid(sample, args.data_shape, args.img_type,
                            sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
        tf.summary.image("perturbed images", plot_to_image(
            figure), max_outputs=1, step=0)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    # Build ScoreNet
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():
            model = get_uncompiled_model(args)
    else:
        model = get_uncompiled_model(args)

    # model.compile(optimizer=optimizer, loss=tfk.losses.MeanSquaredError())
    loss_obj = CustomLoss()
    model.compile(optimizer=optimizer, loss=loss_obj)

    # restore
    if args.restore is not None:
        if mirrored_strategy is not None:
            with mirrored_strategy.scope():
                model.load_weights(abs_restore_path)
                print("Model Weights loaded from {}".format(abs_restore_path))
        else:
            model.load_weights(abs_restore_path)
            print("Model Weights loaded from {}".format(abs_restore_path))

    # Set up callbacks
    logdir = os.path.join("logs", "scalars") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=logdir, write_graph=True, update_freq="epoch",
                                                     profile_batch='500,520', embeddings_freq=0, histogram_freq=0)

    def display_generated_samples(epoch, logs):
        if (args.n_epochs < 10) or (epoch % (args.n_epochs // 10) == 0):
            if mirrored_strategy is not None:
                with mirrored_strategy.scope():
                    gen_samples = anneal_langevin_dynamics(args.data_shape, model, 32, sigmas_np)
            else:
                gen_samples = anneal_langevin_dynamics(args.data_shape, model, 32, sigmas_np)

            if args.use_logit:
                gen_samples = 1. / (1 + np.exp(-gen_samples))
                gen_samples = (gen_samples - args.alpha) / (1. - args.alpha)
            if args.img_type == 'melspec':
                gen_samples = gen_samples * args.dataset_maxval
            try:
                figure = image_grid(gen_samples, args.data_shape, args.img_type,
                                    sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
                sample_image = plot_to_image(figure)
                with file_writer.as_default():
                    tf.summary.image("Generated Samples", sample_image, step=epoch)

            except IndexError:
                with train_summary_writer.as_default():
                    tf.summary.text(name="display error",
                                    data="Impossible to display spectrograms because of NaN values",
                                    step=epoch)

        else:
            pass

    gen_samples_callback = tfk.callbacks.LambdaCallback(on_epoch_end=display_generated_samples)
    # earlystopping_callback = tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

    callbacks = [
        tensorboard_callback,
        tfk.callbacks.ModelCheckpoint(filepath="tf_ckpts/ckpt.{epoch:02d}",
                                      save_weights_only=True,
                                      monitor="loss",
                                      save_freq=42210),
        tfk.callbacks.TerminateOnNaN(),
        gen_samples_callback
    ]

    # Display parameters
    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with file_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)

    # Train
    t0 = time.time()
    model.fit(train_dataset, epochs=args.n_epochs, validation_data=eval_dataset,
              callbacks=callbacks, verbose=2, validation_freq=10)

    model.save_weights("save_weights")

    total_trainable_variables = utils.total_trainable_variables(model)
    print("Total Trainable Variables: ", total_trainable_variables)

    print("Training time: ", np.round(time.time() - t0, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NCSN model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")
    parser.add_argument("--use_logit", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument('--mirrored_strategy', action="store_false")

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--fmin", type=int, default=125)
    parser.add_argument("--fmax", type=int, default=7600)

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
