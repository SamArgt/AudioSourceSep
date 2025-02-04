import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import shutil
import datetime
import matplotlib.pyplot as plt
import io
import os
import librosa
from librosa.display import specshow
import argparse
import yaml
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


"""
Miscelleneaous functions for training
"""


def setUp_optimizer(mirrored_strategy, args):
    lr = args.learning_rate
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():
            if args.optimizer == 'adam':
                optimizer = tfk.optimizers.Adam(lr=lr)
            elif args.optimizer == 'adamax':
                optimizer = tfk.optimizers.Adamax(lr=lr)
            else:
                raise ValueError("optimizer argument should be adam or adamax")
    else:
        if args.optimizer == 'adam':
            optimizer = tfk.optimizers.Adam(lr=lr)
        elif args.optimizer == 'adamax':
            optimizer = tfk.optimizers.Adamax(lr=lr)
        else:
            raise ValueError("optimizer argument should be adam or adamax")

    return optimizer


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


def setUp_checkpoint(mirrored_strategy, model, optimizer, max_to_keep=5, path='./tf_ckpts'):

    # Checkpoint object
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():
            ckpt = tf.train.Checkpoint(
                variables=model.variables, optimizer=optimizer)
            manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_to_keep)
    else:
        ckpt = tf.train.Checkpoint(
            variables=model.variables, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_to_keep)

    return ckpt, manager


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(sample, data_shape, data_type="image", **kwargs):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    if data_shape[-1] == 1:
        sample = np.squeeze(sample, axis=-1)
    for i, ax in enumerate(axes):
        if i > (len(sample) - 1):
            return f
        if data_type == "image":
            ax.imshow(sample[i])
            ax.set_axis_off()
        elif data_type == "melspec":
            specshow(sample[i], sr=kwargs["sampling_rate"],
                     ax=ax, x_axis='off', y_axis='off', fmin=kwargs["fmin"], fmax=kwargs["fmax"])

    return f


def get_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    return new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
