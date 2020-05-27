import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from flow_models import flow_tfk_layers
from flow_models import flow_glow
from flow_models import flow_real_nvp
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


def main():

    parser = argparse.ArgumentParser(
        description='Test Flow model on MNIST dataset')
    parser.add_argument('WEIGHTS', type=str, default=None,
                        help='path to saved weights')
    args = parser.parse_args()

    tf.keras.backend.clear_session()

    batch_size = 256
    K = 16
    n_filters_base = 256

    # Construct a tf.data.Dataset
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: x / 255.)
    ds = ds.shuffle(1024).batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]

    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)
    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet

    bijector = flow_glow.GlowBijector_2blocks(
        K, data_shape, shift_and_log_scale_layer, n_filters_base, minibatch)
    inv_bijector = tfb.Invert(bijector)

    params_str = 'Glow Bijector 2 Blocks: \n\t K = {} \n\t ShiftAndLogScaleResNet \n\t n_filters = {} \n\t batch size = {}'.format(
        K, n_filters_base, batch_size)
    print(params_str)

    flow = tfd.TransformedDistribution(tfd.Normal(
        0., 1.), inv_bijector, event_shape=base_distr_shape)
    print("flow sample shape: ", flow.sample(1).shape)
    print("Total Trainable Variables: ", utils.total_trainable_variables(flow))

    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(variables=flow.variables, optimizer=optimizer)
    ckpt.restore(args.WEIGHTS)
    print("Weights restored from {}".format(args.WEIGHTS))

    ds_val = tfds.load('mnist', split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    ds_val = ds_val.map(lambda x: x / 255.)
    ds_val = ds_val.shuffle(1024).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    t0 = time.time()
    val_loss = tf.keras.metrics.Mean()
    for elt in ds_val:
        val_loss.update(-tf.reduce_mean(flow.log_prob(elt)))
    print('Validation loss = {}'.format(val_loss.result()))
    print("Computed in {} seconds".format(round(time.time() - t0, 3)))


if __name__ == "__main__":
    main()
