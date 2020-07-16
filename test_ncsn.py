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

BUFFER_SIZE = 10000
BATCH_SIZE = 512
alpha = 0.05

data_shape = [32, 32, 1]


def get_uncompiled_model():
    # inputs
    perturbed_X = tfk.Input(shape=data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network.CondRefineNetDilated(data_shape, 64, 10, True)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name="ScoreNetwork")

    return model


class CustomLoss(tfk.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def __call__(self, scores, target, sample_weight=None):
        loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
        if sample_weight is not None:
            return tf.reduce_mean(loss * sample_weight)
        else:
            return tf.reduce_mean(loss)


print("Start Testing...")

datasets = tfds.load(name='mnist', with_info=False, as_supervised=False)
ds_train, ds_test = datasets['train'], datasets['test']
dataset_maxval = 256.

sigmas_np = np.logspace(0., -2., num=10)
sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)


def preprocess(x):
    sigma_idx = tf.random.uniform(shape=(), maxval=10, dtype=tf.int32)
    used_sigma = tf.gather(params=sigmas_tf, indices=sigma_idx)
    X = tf.cast(x['image'], tf.float32)
    X = tf.pad(X, tf.constant([[2, 2], [2, 2], [0, 0]]))
    X = (X + tf.random.uniform(data_shape)) / dataset_maxval
    X = X * (1. - alpha) + alpha
    X = tf.math.log(X) - tf.math.log(1. - X)
    perturbed_X = X + tf.random.normal(data_shape) * used_sigma
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

print('Data Loaded...')

model = get_uncompiled_model()
optimizer = tfk.optimizers.Adam()
loss_obj = CustomLoss()
model.compile(optimizer=optimizer, loss=loss_obj)


