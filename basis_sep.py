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

def load_data(args):

    if args.dataset == 'mnist':
        data_shape = (28, 28, 1)
    elif args.dataset == 'cifar10':
        data_shape = (32, 32, 3)
    else:
        raise ValueError("args.dataset should be mnist or cifar10")

    global_batch_size = args.batch_size
    ds = tfds.load(args.dataset, split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    if args.use_logit:
        ds = ds.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
        ds = ds.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds = ds.map(lambda x: x / 256. - 0.5)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
    ds = ds.batch(global_batch_size, drop_remainder=True)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    # ds_dist = mirrored_strategy.experimental_distribute_dataset(ds)
    # Validation Set
    ds_val = tfds.load(args.dataset, split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    ds_val = ds_val.map(
        lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
    if args.use_logit:
        ds_val = ds_val.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
        ds_val = ds_val.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds_val = ds_val.map(lambda x: x / 256. - 0.5)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
    ds_val = ds_val.batch(5000)
    # ds_val_dist = mirrored_strategy.experimental_distribute_dataset(ds_val)

    return ds, ds_val, minibatch


def get_mixture_dataset(args):
    ds, ds_val, minibatch = load_data(args)
    ds1 = ds.shuffle(2048, seed=42, reshuffle_each_iteration=False)
    ds2 = ds.shuffle(2048, seed=84, reshuffle_each_iteration=False)
    ds_zip = tf.data.Dataset.zip((ds1, ds2))
    ds_mix = ds_zip.map(lambda x, y: (x + y) / 2.)
    return tf.data.Dataset.zip((ds1, ds2, ds_mix)), minibatch

def get_mixture(args):

    if args.dataset == 'mnist':
        data_shape = (28, 28, 1)
    elif args.dataset == 'cifar10':
        data_shape = (32, 32, 3)
    else:
        raise ValueError("args.dataset should be mnist or cifar10")

    ds_mix, minibatch = get_mixture_dataset(args)
    gt1, gt2, mixed = list(ds_mix.as_numpy_iterator())[0]
    gt1 = gt1[0]
    gt2 = gt2[0]
    mixed = mixed[0]

    x1 = tf.random.uniform(data_shape, minval=-.5, maxval=.5)
    x2 = tf.random.uniform(data_shape, minval=-.5, maxval=.5)

    return mixed, x1, x2, gt1, gt2, minibatch


def build_flow(args, minibatch):
    tfk.backend.clear_session()

    # Set flow parameters
    if args.dataset == 'mnist':
        data_shape = [28, 28, 1]
    elif args.dataset == 'cifar10':
        data_shape = [32, 32, 3]
    if args.L == 2:
        base_distr_shape = [data_shape[0] // 4, data_shape[1] // 4, data_shape[3] * 16]
    elif args.L == 3:
        base_distr_shape = [data_shape[0] // 8, data_shape[1] // 8, data_shape[3] * 32]
    else:
        raise ValueError("L should be 2 or 3")

    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet

    # Build Flow and Optimizer
    if args.L == 2:
        bijector = flow_glow.GlowBijector_2blocks(args.K, data_shape,
                                                  shift_and_log_scale_layer,
                                                  args.n_filters, minibatch, **args.l2_reg)
    elif args.L == 3:
        bijector = flow_glow.GlowBijector_3blocks(args.K, data_shape,
                                                  shift_and_log_scale_layer,
                                                  args.n_filters, minibatch, **args.l2_reg)
    inv_bijector = tfb.Invert(bijector)
    flow = tfd.TransformedDistribution(tfd.Normal(
        0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow


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

def basis_inner_loop(mixed, x1, x2, model1, model2, sigma, args):

    if args.dataset == 'mnist':
        data_shape = [28, 28, 1]
    elif args.dataset == 'cifar10':
        data_shape == [32, 32, 3]

    eta = .00003 * (sigma / .01) ** 2
    lambda_recon = 1.0 / (sigma ** 2)
    for t in range(args.T):

        epsilon1 = tf.math.sqrt(2 * eta) * tf.random.normal(data_shape)
        epsilon2 = tf.math.sqrt(2 * eta) * tf.random.normal(data_shape)

        grad_logprob1 = compute_grad_logprob(x1, model1)
        grad_logprob2 = compute_grad_logprob(x2, model2)
        x1 = x1 + eta * (grad_logprob1 - lambda_recon * (x1 + x2 - 2 * mixed)) + epsilon1
        x2 = x2 + eta * (grad_logprob2 - lambda_recon * (x1 + x2 - 2 * mixed)) + epsilon2

    return x1, x2


def basis_outer_loop(restore_dict, args, train_summary_writer):

    mixed, x1, x2, gt1, gt2, minibatch = get_mixture(args)

    with train_summary_writer.as_default():
        tf.summary.image("Mix and originals", np.array([mixed, gt1, gt2], step=0))

    model = build_flow(args, minibatch)

    optimizer = setUp_optimizer(args)
    step = 0
    for sigma, restore_path in restore_dict.items():
        print("Sigma = {}".format(sigma))
        restore_checkpoint(restore_path, args, model, optimizer)
        print("Model restored")
        model1 = model2 = model

        x1, x2 = basis_inner_loop(mixed, x1, x2, model1, model2, sigma, args)
        step += 1

        with train_summary_writer.as_default():
            tf.summary.image("Components", np.array([x1, x2]), max_outputs=10, step=step)

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
        description='Train Flow model on MNIST dataset')
    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved models')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='mnist_trained_flow',
                        help='output dirpath for savings')

    # BASIS hyperparameters
    parser.add_argument("--T", type=int, default=100,
                        help="Number of iteration in the inner loop")

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

    args = parser.parse_args()

    main(args)
