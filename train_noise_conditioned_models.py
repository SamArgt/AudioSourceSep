import train_flow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import utils
from flow_models import flow_builder
from pipeline import data_loader
import argparse
import os
import shutil
import datetime
import sys
import io
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def setUp_optimizer(mirrored_strategy, args):
    lr = args.learning_rate
    with mirrored_strategy.scope():
        if args.optimizer == 'adam':
            optimizer = tfk.optimizers.Adam(
                lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
        elif args.optimizer == 'adamax':
            optimizer = tfk.optimizers.Adamax(lr=lr)
        else:
            raise ValueError("optimizer argument should be adam or adamax")
    return optimizer


def setUp_tensorboard():
    # Tensorboard
    # Clear any logs from previous runs
    try:
        shutil.rmtree('tensorboard_logs')
    except FileNotFoundError:
        pass
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(
        'tensorboard_logs', 'gradient_tape', current_time, 'train')
    test_log_dir = os.path.join(
        'tensorboard_logs', 'gradient_tape', current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    return train_summary_writer, test_summary_writer


def setUp_checkpoint(mirrored_strategy, args, flow, optimizer):

    # Checkpoint object
    with mirrored_strategy.scope():
        ckpt = tf.train.Checkpoint(
            variables=flow.variables, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)

    return ckpt, manager


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(sample):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(sample[i] + 0.5, 0., 1.))
        ax.set_axis_off()
    return f


def main(args):

    sigmas = np.logspace(np.log(args.sigmaL) / np.log(10), np.log(args.sigma1) / np.log(10), num=args.n_sigmas)
    abs_restore_path = os.path.abspath(args.RESTORE)

    if args.output == 'noise_conditioned_flows':
        output_dirname = 'glow_' + args.dataset + '_' + str(args.L) + '_' + \
            str(args.K) + '_' + str(args.n_filters) + \
            '_' + str(args.batch_size)
        output_dirpath = os.path.join(args.output, output_dirname)
    else:
        output_dirpath = args.output

    output_dirpath = os.path.abspath(output_dirpath)
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
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))

    # Load Dataset
    ds, _, ds_dist, ds_val_dist, minibatch = data_loader.load_data(dataset=args.dataset, batch_size=args.batch_size,
                                                                   use_logit=args.use_logit, alpha=args.alpha,
                                                                   noise=args.noise, mirrored_strategy=mirrored_strategy)

    # Build Flow and Set up optimizer
    flow = flow_builder.build_flow(minibatch, L=args.L, K=args.K, n_filters=args.n_filters, dataset=args.dataset,
                                   l2_reg=args.l2_reg, mirrored_strategy=mirrored_strategy, learntop=args.learntop)

    params_dict = vars(args)
    template = 'Glow Flow \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with mirrored_strategy.scope():
        print("flow sample shape: ", flow.sample(1).shape)

    total_trainable_variables = utils.total_trainable_variables(flow)
    print("Total Trainable Variables: ", total_trainable_variables)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    for sigma in sigmas:
        os.chdir(output_dirpath)
        try:
            os.mkdir('sigma_{}'.format(round(sigma, 2)))
            os.chdir('sigma_{}'.format(round(sigma, 2)))
        except FileExistsError:
            os.chdir('sigma_{}'.format(round(sigma, 2)))

        print("_" * 100)
        print("Training at noise level {}".format(round(sigma, 2)))
        # Set up tensorboard
        train_summary_writer, test_summary_writer = setUp_tensorboard()

        # Set up checkpoint
        ckpt, manager = setUp_checkpoint(
            mirrored_strategy, args, flow, optimizer)

        # restore
        with mirrored_strategy.scope():
            status = ckpt.restore(abs_restore_path)
            assert optimizer.iterations > 0
            status.assert_existing_objects_matched()
            print("Model Restored from {}".format(abs_restore_path))

        # Set up optimizer
        # optimizer = setUp_optimizer(mirrored_strategy, args)

        # load noisy data
        args.noise = sigma
        ds, _, ds_dist, ds_val_dist, minibatch = data_loader.load_data(dataset='mnist', batch_size=args.batch_size,
                                                                       use_logit=args.use_logit, alpha=args.alpha,
                                                                       noise=args.noise, mirrored_strategy=mirrored_strategy)
        with train_summary_writer.as_default():
            sample = list(ds.take(1).as_numpy_iterator())[0]
            sample = sample[:32]
            figure = image_grid(sample)
            tf.summary.image("original images", plot_to_image(figure), max_outputs=1, step=0)

        params_dict = vars(args)
        template = 'Glow Flow \n\t '
        for k, v in params_dict.items():
            template += '{} = {} \n\t '.format(k, v)
        with train_summary_writer.as_default():
            tf.summary.text(name='Parameters',
                            data=tf.constant(template), step=0)
            tf.summary.text(name="Total Trainable Variables",
                            data=tf.constant(str(total_trainable_variables)), step=0)

        # Train
        training_time, save_path = train_flow.train(mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
                                                    manager, None, train_summary_writer, test_summary_writer)
        print("Training time: ", np.round(training_time, 2), ' seconds')

        # Fine tune model serially
        abs_restore_path = os.path.abspath(save_path)
        args.RESTORE = abs_restore_path

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train noise conditioned Flow Model')

    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='noise_conditioned_flows',
                        help='output dirpath for savings')
    parser.add_argument('--debug', action="store_true")

    # Noise parameters
    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigmaL', type=float, default=0.01)
    parser.add_argument('--n_sigmas', type=float, default=10)

    # Model hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=512,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    # preprocessing parameters
    parser.add_argument('--use_logit', action="store_true",
                        help="Either to use logit function to preprocess the data")
    parser.add_argument('--alpha', type=float, default=10**(-6),
                        help='preprocessing parameter: x = logit(alpha + (1 - alpha) * z / 256.). Only if use logit')
    parser.add_argument('--noise', type=float, default=None,
                        help='noise level for BASIS separation')

    args = parser.parse_args()

    main(args)
