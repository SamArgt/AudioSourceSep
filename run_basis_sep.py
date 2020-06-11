import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import flow_builder
from pipeline import data_loader
from basis_sep import basis_sep
import argparse
import time
import os
import sys
import shutil
import datetime
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def setUp_optimizer(args):
    lr = args.learning_rate
    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam(lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax(lr=lr)
    else:
        raise ValueError("optimizer argument should be adam or adamax")
    return optimizer


def restore_checkpoint(restore_path, args, flow, optimizer):
    restore_abs_dirpath = os.path.abspath(restore_path)

    # Checkpoint object
    ckpt = tf.train.Checkpoint(
        variables=flow.variables, optimizer=optimizer)
    # Restore weights if specified
    ckpt.restore(restore_abs_dirpath)

    return ckpt


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


@tf.function
def compute_grad_logprob(X, model):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = -tf.reduce_mean(model.log_prob(X))
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


def basis_outer_loop(restore_dict, args, train_summary_writer):

    mixed, x1, x2, gt1, gt2, minibatch = data_loader.get_mixture(dataset=args.dataset, batch_size=args.batch_size,
                                                                 use_logit=args.use_logit, alpha=args.alpha, noise=args.noise,
                                                                 mirrored_strategy=None)

    with train_summary_writer.as_default():
        tf.summary.image("Mix and originals",
                         np.concatenate((mixed[:5], gt1[:5], gt2[:5]), axis=0), step=0)

    model = flow_builder.build_flow(minibatch, L=args.L, K=args.K, n_filters=args.n_filters, dataset=args.dataset,
                                    l2_reg=args.l2_reg, mirrored_strategy=None)

    optimizer = setUp_optimizer(args)
    step = 0
    for sigma, restore_path in restore_dict.items():
        print("Sigma = {}".format(sigma))
        restore_checkpoint(restore_path, args, model, optimizer)
        print("Model restored")
        model1 = model2 = model

        x1, x2 = basis_sep.basis_inner_loop(mixed, x1, x2, model1, model2, sigma, args.n_mixed, sigmaL=args.sigmaL,
                                            delta=2e-5, T=100, dataset=args.dataset)
        step += 1

        with train_summary_writer.as_default():
            tf.summary.image("Components", np.concatenate((x1[:5], x2[:5]), axis=0), max_outputs=10, step=step)

        print("inner loop done")
        print("_" * 100)

    return mixed, x1, x2, gt1, gt2


def main(args):

    try:
        os.mkdir(args.output)
        os.chdir(args.output)
    except FileExistsError:
        os.chdir(args.output)

    log_file = open('out.log', 'w')
    sys.stdout = log_file

    train_summary_writer, test_summary_writer = setUp_tensorboard()

    sigmas = np.linspace(0.01, 1., num=10)
    sigmas = np.flip(sigmas)
    restore_dict = {sigma: args.RESTORE + '_' + str(sigma) for sigma in sigmas}
    t0 = time.time()
    mixed, x1, x2, gt1, gt2 = basis_outer_loop(restore_dict, args, train_summary_writer)
    t1 = time.time()

    x1 = x1.numpy()
    x2 = x2.numpy()
    np.savez('results', x1=x1, x2=x2, gt1=gt1, gt2=gt2, mixed=mixed)

    print("Duration: {} seconds".format(round(t1 - t0, 3)))

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BASIS Separatation')
    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved models')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='basis_sep',
                        help='output dirpath for savings')

    # BASIS hyperparameters
    parser.add_argument("--T", type=int, default=100,
                        help="Number of iteration in the inner loop")
    parser.add_argument('--n_mixed', type=int, default=10,
                        help="number of mixture to separate")

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
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
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

    args = parser.parse_args()

    main(args)
