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

    data_shape = (32, 32, 3)

    buffer_size = 2048
    global_batch_size = args.batch_size
    ds = tfds.load(args.dataset, split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    if args.dataset == 'mnist':
        ds = ds.map(lambda x: tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]])))
    if args.use_logit:
        ds = ds.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
        ds = ds.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds = ds.map(lambda x: x / 256. - 0.5)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
    ds = ds.shuffle(buffer_size).batch(global_batch_size, drop_remainder=True)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    # Validation Set
    ds_val = tfds.load(args.dataset, split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    if args.dataset == 'mnist':
        ds_val = ds_val.map(lambda x: tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]])))
    if args.use_logit:
        ds_val = ds_val.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
        ds_val = ds_val.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds_val = ds_val.map(lambda x: x / 256. - 0.5)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
    ds_val = ds_val.batch(5000)

    return ds, ds_val, minibatch


def build_flow(args, minibatch):
    tfk.backend.clear_session()

    # Set flow parameters
    data_shape = [32, 32, 3]
    if args.L == 2:
        base_distr_shape = [data_shape[0] // 4, data_shape[1] // 4, data_shape[2] * 16]
    elif args.L == 3:
        base_distr_shape = [data_shape[0] // 8, data_shape[1] // 8, data_shape[2] * 32]
    else:
        raise ValueError("L should be 2 or 3")

    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet

    # Build Flow and Optimizer
    if args.L == 2:
        bijector = flow_glow.GlowBijector_2blocks(args.K, data_shape,
                                                  shift_and_log_scale_layer,
                                                  args.n_filters, minibatch, **{'l2_reg': args.l2_reg})
    elif args.L == 3:
        bijector = flow_glow.GlowBijector_3blocks(args.K, data_shape,
                                                  shift_and_log_scale_layer,
                                                  args.n_filters, minibatch, **{'l2_reg': args.l2_reg})
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


def restore_checkpoint(args_parsed, flow, optimizer):
    restore_abs_dirpath = os.path.abspath(args_parsed.RESTORE)

    # Checkpoint object
    ckpt = tf.train.Checkpoint(
        variables=flow.variables, optimizer=optimizer)
    # Restore weights if specified
    ckpt.restore(restore_abs_dirpath)

    return ckpt


def evaluate(args_parsed, flow, ds, ds_val):

    D = tf.constant(28 * 28 * 1, dtype=tf.float32)  # dimension of the data

    @tf.function
    def eval_step(inputs):
        losses = flow.log_prob(inputs)

        log_lik = losses
        if args_parsed.use_logit:
            log_lik += D * tf.math.log((1 - args_parsed.alpha) / 256.)
            log_lik -= tf.reduce_sum(tf.math.sigmoid(inputs) * (1 - tf.math.sigmoid(inputs)), axis=[1, 2, 3])
        else:
            log_lik += D * tf.math.log(1. / 256.)

        bits_per_pixel = - log_lik / (D * tf.math.log(2.))

        avg_loss = -tf.reduce_mean(losses)
        avg_neg_log_lik = -tf.reduce_mean(log_lik)
        avg_bits_per_pixel = tf.reduce_mean(bits_per_pixel)

        return avg_loss, avg_neg_log_lik, avg_bits_per_pixel

    train_loss = tfk.metrics.Mean(name='train loss')
    train_nll = tfk.metrics.Mean(name="train nll")
    train_bits_per_pixel = tfk.metrics.Mean(name="train bits per pixel")

    test_loss = tfk.metrics.Mean(name='test loss')
    test_nll = tfk.metrics.Mean(name="test nll")
    test_bits_per_pixel = tfk.metrics.Mean(name="test bits per pixel")

    for batch in ds:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch)

        train_loss.update_state(avg_loss)
        train_nll.update_state(avg_neg_log_lik)
        train_bits_per_pixel.update_state(avg_bits_per_pixel)

    for batch in ds_val:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch)

        test_loss.update_state(avg_loss)
        test_nll.update_state(avg_neg_log_lik)
        test_bits_per_pixel.update_state(avg_bits_per_pixel)

    return train_loss, train_nll, train_bits_per_pixel, test_loss, test_nll, test_bits_per_pixel


def main(args_parsed):

    result_file = open(args_parsed.output, "a")
    sys.stdout = result_file

    ds, ds_val, minibatch = load_data(args_parsed)

    flow = build_flow(args_parsed, minibatch)

    optimizer = setUp_optimizer(args_parsed)

    restore_checkpoint(args_parsed, flow, optimizer)

    print('_' * 100)

    params_dict = vars(args_parsed)
    template = 'Glow Flow \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    print("Total Trainable Variables: ", utils.total_trainable_variables(flow))

    print("\nWeights restored from {} \n".format(args_parsed.RESTORE))

    print('Start Evaluation...\n')
    t0 = time.time()
    tfk_metrics = evaluate(args_parsed, flow, ds, ds_val)
    t1 = time.time()

    for m in tfk_metrics:
        print("{}: {}".format(m.name, m.result()))

    print('Duration: {}'.format(round(t1 - t0, 3)))
    print('_' * 100)
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model on MNIST dataset')
    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='evaluation.txt',
                        help='output dirpath for savings')

    # Model hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=512,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")

    # Optimization parameters
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

    args_parsed = parser.parse_args()

    main(args_parsed)
