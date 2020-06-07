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
    buffer_size = 2048
    global_batch_size = args.batch_size
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: x + tf.random.uniform(shape=(28, 28, 1),
                                                minval=0., maxval=1. / 256.))
    ds = ds.map(lambda x: x / 256.)
    if args.use_logit:
        ds = ds.map(lambda x: args.alpha + (1 - args.alpha) * x)
        ds = ds.map(lambda x: tf.math.log(x / (1 - x)))
    ds = ds.shuffle(buffer_size).batch(global_batch_size)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    # Validation Set
    ds_val = tfds.load('mnist', split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    ds_val = ds_val.map(
        lambda x: x + tf.random.uniform(shape=(28, 28, 1), minval=0., maxval=1. / 256.))
    ds_val = ds_val.map(lambda x: x / 256.)
    if args.use_logit:
        ds_val = ds_val.map(lambda x: args.alpha + (1 - args.alpha) * x)
        ds_val = ds_val.map(lambda x: tf.math.log(x / (1 - x)))
    ds_val = ds_val.batch(5000)

    return ds, ds_val, minibatch


def build_flow(args, minibatch):
    tfk.backend.clear_session()

    # Set flow parameters
    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)
    K = args.K
    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet
    n_filters_base = args.n_filters

    # Build Flow and Optimizer
    bijector = flow_glow.GlowBijector_2blocks(K, data_shape,
                                              shift_and_log_scale_layer, n_filters_base, minibatch)
    inv_bijector = tfb.Invert(bijector)
    flow = tfd.TransformedDistribution(tfd.Normal(
        0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow


def setUp_optimizer(args):
    lr = args.learning_rate
    optimizer = tfk.optimizers.Adam(lr=lr)
    return optimizer


def restore_checkpoint(args, flow, optimizer):
    restore_abs_dirpath = os.path.abspath(args.restore)

    # Checkpoint object
    ckpt = tf.train.Checkpoint(
        variables=flow.variables, optimizer=optimizer)
    # Restore weights if specified
    ckpt.restore(restore_abs_dirpath)

    return ckpt


def evaluate(args, flow, ds, ds_val):

    D = 28 * 28 * 1  # dimension of the data

    @tf.function
    def eval_step(inputs, args):
        losses = flow.log_prob(inputs)

        log_lik = losses
        if args.use_logit:
            log_lik += D * tf.math.log((1 - args.alpha) / 256.)
            log_lik -= tf.reduce_sum(tf.math.sigmoid(inputs) * (1 - tf.math.sigmoid(inputs)), axis=[1, 2, 3])
        else:
            log_lik += D * tf.math.log(1. / 256.)

        bits_per_pixel = - log_lik / (D * tf.math.log(2))

        avg_loss = - tf.reduce_mean(losses)
        avg_neg_log_lik = - tf.reduce_mean(log_lik)
        avg_bits_per_pixel = -tf.reduce_mean(bits_per_pixel)

        return avg_loss, avg_neg_log_lik, avg_bits_per_pixel

    train_loss = tfk.metrics.Mean(name='train loss')
    train_nll = tfk.metrics.Mean(name="train nll")
    train_bits_per_pixel = tfk.metrics.Mean(name="train bits per pixel")

    test_loss = tfk.metrics.Mean(name='test loss')
    test_nll = tfk.metrics.Mean(name="test nll")
    test_bits_per_pixel = tfk.metrics.Mean(name="test bits per pixel")

    for batch in ds:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch, args)

        train_loss.update_state(avg_loss)
        train_nll.update_state(avg_neg_log_lik)
        train_bits_per_pixel.update_state(avg_bits_per_pixel)

    for batch in ds_val:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch, args)

        test_loss.update_state(avg_loss)
        test_nll.update_state(avg_neg_log_lik)
        test_bits_per_pixel.update_state(avg_bits_per_pixel)

    return train_loss, train_nll, train_bits_per_pixel, test_loss, test_nll, test_bits_per_pixel


def main(args):

    result_file = open(args.output, "a")
    sys.stdout = result_file

    print('_' * 100)

    params_dict = vars(args)
    template = 'Glow Flow \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    ds, ds_val, minibatch = load_data(args)

    flow = build_flow(args, minibatch)

    optimizer = setUp_optimizer(args)

    restore_checkpoint(args, flow, optimizer)

    print('Start Evaluation...')
    t0 = time.time()
    tfk_metrics = evaluate(args, flow, ds, ds_val)
    t1 = time.time()

    for m in tfk_metrics:
        print("{}: {}".format(m.name, m.result()))

    print('Duration: {}'.format(round(t1 - t0, 3)))
    print('_' * 100)
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Flow model on MNIST dataset')
    parser.add_argument('RESTORE', type=str,
                        help='directory of saved weights')
    parser.add_argument('--output', type=str, default='results.txt',
                        help="File where to save the results")
    parser.add_argument('--K', type=int, default=16,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_filters', type=int, default=256,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--use_logit', action="store_true",
                        help="Either to use logit function to preprocess the data")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=10**(-6),
                        help='preprocessing parameter: x = logit(alpha + (1 - alpha) * z / 256.)')
    args = parser.parse_args()

    main(args)
