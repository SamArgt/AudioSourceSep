import train_flow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from flow_models import flow_tfk_layers
from flow_models import flow_glow
from flow_models import flow_real_nvp
from flow_models import flow_tfp_bijectors
from flow_models import utils
import argparse
import time
import os
import sys
import shutil
import datetime
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def main(args):

    sigmas = np.linspace(args.sigma1, args.sigmaL, num=args.n_sigmas)
    abs_restore_path = os.path.abspath(args.RESTORE)

    if args.output == 'mnist_noise_conditioned':
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
    sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))

    # Load Dataset
    ds_dist, ds_val_dist, minibatch = train_flow.load_data(
        mirrored_strategy, args)

    # Build Flow and Set up optimizer
    flow = train_flow.build_flow(mirrored_strategy, args, minibatch)

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
    optimizer = train_flow.setUp_optimizer(mirrored_strategy, args)

    for sigma in sigmas:
        os.chdir(output_dirpath)
        try:
            os.mkdir('sigma_{}'.format(sigma))
            os.chdir('sigma_{}'.format(sigma))
        except FileExistsError:
            os.chdir('sigma_{}').format(sigma)

        print("_" * 100)
        print("Training at noise level {}".format(sigma))
        # Set up tensorboard
        train_summary_writer, test_summary_writer = train_flow.setUp_tensorboard()

        # Set up checkpoint
        ckpt, manager, manager_issues = train_flow.setUp_checkpoint(
            mirrored_strategy, args, flow, optimizer)

        # restore
        with mirrored_strategy.scope():
            ckpt.restore(abs_restore_path)
            print("Model Restored")

        # load noisy data
        args.noise = sigma
        ds_dist, ds_val_dist, minibatch = train_flow.load_data(
            mirrored_strategy, args)

        with train_summary_writer.as_default():
            tf.summary.text(name='Parameters',
                            data=tf.constant(template), step=0)
            tf.summary.text(name="Total Trainable Variables",
                            data=tf.constant(str(total_trainable_variables)), step=0)

        # Train
        training_time = train_flow.train(mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
                                         manager, manager_issues, train_summary_writer, test_summary_writer)
        print("Training time: ", np.round(training_time, 2), ' seconds')

        # Saving the last variables
        save_path = manager.save()
        print("Model Saved at {}".format(save_path))

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model on MNIST dataset')

    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='mnist_noise_conditioned',
                        help='output dirpath for savings')

    # Noise parameters
    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigmaL', type=float, default=0.01)
    parser.add_argument('n_sigmas', type=float, default=10)

    # Model hyperparameters
    parser.add_argument('--L', default=2, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=16,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=256,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")

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
