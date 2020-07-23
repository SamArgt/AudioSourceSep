import tensorflow as tf
import tensorflow_datasets as tfds
from .preprocessing import load_tf_records
import os
import re


def load_toydata(dataset='mnist', batch_size=256, mirrored_strategy=None, reshuffle=True):

    buffer_size = 2048
    global_batch_size = batch_size
    ds = tfds.load(dataset, split='train', shuffle_files=True)
    ds_val = tfds.load(dataset, split='test', shuffle_files=True)
    ds = ds.map(lambda x: tf.cast(x['image'], tf.float32))
    ds_val = ds_val.map(lambda x: tf.cast(x['image'], tf.float32))

    if dataset == 'mnist':
        ds = ds.map(lambda x: tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]])))
        ds_val = ds_val.map(lambda x: tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]])))

    ds = ds.shuffle(buffer_size, reshuffle_each_iteration=reshuffle)
    ds = ds.batch(global_batch_size, drop_remainder=True)
    minibatch = list(ds.take(1))[0]

    ds_val = ds_val.batch(5000)

    if mirrored_strategy is not None:
        ds_dist = mirrored_strategy.experimental_distribute_dataset(ds)
        ds_val_dist = mirrored_strategy.experimental_distribute_dataset(ds_val)
        return ds, ds_val, ds_dist, ds_val_dist, minibatch

    else:
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        return ds, ds_val, minibatch


def get_mixture(dataset='mnist', n_mixed=10, use_logit=False, alpha=None, noise=0.1, mirrored_strategy=None):

    if dataset == 'mnist':
        data_shape = [n_mixed, 32, 32, 1]
    elif dataset == 'cifar10':
        data_shape = [n_mixed, 32, 32, 3]
    else:
        raise ValueError("args.dataset should be mnist or cifar10")

    ds, _, minibatch = load_toydata(dataset, n_mixed, use_logit, alpha, noise, mirrored_strategy, preprocessing=False)

    ds1 = ds.take(1)
    ds2 = ds.take(1)
    for gt1, gt2 in zip(ds1, ds2):
        gt1, gt2 = gt1, gt2

    gt1 = gt1 / 256. - .5 + tf.random.uniform(data_shape, minval=0., maxval=1. / 256.)
    gt2 = gt2 / 256. - .5 + tf.random.uniform(data_shape, minval=0., maxval=1. / 256.)
    mixed = (gt1 + gt2) / 2.

    # x1 = tf.random.uniform(data_shape, minval=-.5, maxval=.5)
    # x2 = tf.random.uniform(data_shape, minval=-.5, maxval=.5)
    x1 = tf.random.normal(data_shape)
    x2 = tf.random.normal(data_shape)

    return mixed, x1, x2, gt1, gt2, minibatch


def load_melspec_ds(train_dirpath, test_dirpath, batch_size=256, reshuffle=True, mirrored_strategy=None):

    train_melspec_files = []
    train_dirpath = os.path.abspath(train_dirpath)
    for root, dirs, files in os.walk(train_dirpath):
        current_path = os.path.join(train_dirpath, root)
        if len(files) > 0:
            train_melspec_files += [os.path.join(current_path, f) for f in files if re.match(".*(.)tfrecord$", f)]

    test_melspec_files = []
    test_dirpath = os.path.abspath(test_dirpath)
    for root, dirs, files in os.walk(test_dirpath):
        current_path = os.path.join(test_dirpath, root)
        if len(files) > 0:
            test_melspec_files += [os.path.join(current_path, f) for f in files if re.match(".*(.)tfrecord$", f)]

    buffer_size = 2048
    ds_train = load_tf_records(train_melspec_files)
    ds_train = ds_train.shuffle(buffer_size, reshuffle_each_iteration=False)
    ds_train = ds_train.map(lambda x: tf.expand_dims(x, axis=-1))
    n_train = len(list(ds_train.as_numpy_iterator()))

    ds_test = load_tf_records(test_melspec_files)
    ds_test = ds_test.shuffle(buffer_size, reshuffle_each_iteration=False)
    ds_test = ds_test.map(lambda x: tf.expand_dims(x, axis=-1))
    n_test = len(list(ds_test.as_numpy_iterator()))

    if batch_size is not None:

        ds_train = ds_train.batch(batch_size, drop_remainder=True)
        ds_test = ds_test.batch(batch_size, drop_remainder=True)

    minibatch = list(ds_train.take(1).as_numpy_iterator())[0]

    if mirrored_strategy is not None:
        ds_train_dist = mirrored_strategy.experimental_distribute_dataset(ds_train)
        ds_test_dist = mirrored_strategy.experimental_distribute_dataset(ds_test)
        return ds_train, ds_test, ds_train_dist, ds_test_dist, minibatch, n_train, n_test

    else:
        return ds_train, ds_test, minibatch, n_train, n_test
