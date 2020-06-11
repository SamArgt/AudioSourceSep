import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(dataset='mnist', batch_size=256, use_logit=False, alpha=None, noise=None, mirrored_strategy=None):

    if dataset == 'mnist':
        data_shape = (32, 32, 1)
    elif dataset == 'cifar10':
        data_shape = (32, 32, 3)
    else:
        raise ValueError("dataset should be mnist or cifar10")

    buffer_size = 2048
    global_batch_size = batch_size
    ds = tfds.load(dataset, split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    if dataset == 'mnist':
        ds = ds.map(lambda x: tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]])))
    if use_logit:
        ds = ds.map(lambda x: alpha + (1 - alpha) * x / 256.)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
        ds = ds.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds = ds.map(lambda x: x / 256. - 0.5)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))

    if noise is not None:
        ds = ds.map(lambda x: x + tf.random.normal(shape=data_shape,
                                                   mean=0, stddev=noise))

    ds = ds.shuffle(buffer_size).batch(global_batch_size, drop_remainder=True)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]

    # Validation Set
    ds_val = tfds.load(dataset, split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    if dataset == 'mnist':
        ds_val = ds_val.map(lambda x: tf.pad(
            x, tf.constant([[2, 2], [2, 2], [0, 0]])))
    if use_logit:
        ds_val = ds_val.map(lambda x: alpha + (1 - alpha) * x / 256.)
        ds_val = ds_val.map(
            lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
        ds_val = ds_val.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds_val = ds_val.map(lambda x: x / 256. - 0.5)
        ds_val = ds_val.map(
            lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))

    if noise is not None:
        ds_val = ds_val.map(
            lambda x: x + tf.random.normal(shape=data_shape, mean=0, stddev=noise))
    ds_val = ds_val.batch(5000)

    if mirrored_strategy is not None:
        ds_dist = mirrored_strategy.experimental_distribute_dataset(ds)
        ds_val_dist = mirrored_strategy.experimental_distribute_dataset(ds_val)
        return ds_dist, ds_val_dist, minibatch

    else:
        return ds, ds_val, minibatch
