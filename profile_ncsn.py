import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import flow_builder
from ncsn import score_network
from pipeline import data_loader
from librosa.display import specshow
import librosa
import train_utils
import argparse
import time
import os
import sys
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

data_shape = [96, 64, 1]
num_classes = 10
n_filters = 92


def get_uncompiled_model():
    # inputs
    perturbed_X = tfk.Input(
        shape=data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network.CondRefineNetDilated(data_shape, n_filters,
                                                 num_classes, False)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx],
                      outputs=outputs, name="ScoreNetwork")

    return model


def restore_checkpoint(ckpt, restore_path, model, optimizer, latest=True):
    if latest:
        checkpoint_restore_path = tf.train.latest_checkpoint(restore_path)
        assert restore_path is not None, restore_path
    else:
        checkpoint_restore_path = restore_path
    # Restore weights if specified
    status = ckpt.restore(checkpoint_restore_path)
    status.assert_existing_objects_matched()

    return ckpt


def main(args):
    abs_restore_path = os.path.abspath(args.RESTORE)
    # set up optimizer
    optimizer = tfk.optimizers.Adam()
    # model
    model = get_uncompiled_model()
    # checkpoint
    t0 = time.time()
    ckpt, _, _ = train_utils.setUp_checkpoint(model, optimizer)
    restore_checkpoint(ckpt, abs_restore_path, model, optimizer, latest=False)
    print("Mode restored in {} seconds".format(round(time.time() - t0, 3)))

    t_init = time.time()
    for sigma in range(10):
        print("Sigma index: {}".format(sigma))
        x = tf.random.normal([10, 96, 34, 1])
        sigma_idx = tf.ones_like(x, dtype=tf.int32) * sigma
        t0 = time.time()
        scores = model([x, sigma_idx])
        print("score computed in {} seconds".format(round(time.time() - t0, 3)))
        print('_' * 100)

    print("TOTAL TIME FOR 10 ITERATIONS: {} seconds".format(time.time() - t_init, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile NCSN')
    parser.add_argument("RESTORE")

    args = parser.parse_args()
    main(args)
