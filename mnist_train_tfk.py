import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flow_models import flow_tfk_layers
from flow_models import flow_tfk_models
from flow_models import utils
import argparse
import time
import os
import sys
tfk = tf.keras


def main():

    parser = argparse.ArgumentParser(
        description='Train Flow model on MNIST dataset')
    parser.add_argument('OUTPUT', type=str,
                        help='output dirpath for savings')
    parser.add_argument('N_EPOCHS', type=str,
                        help='number of epochs to train')
    args = parser.parse_args()
    output_dirpath = args.OUTPUT

    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)

    log_file = open('out.log', 'w')
    sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    N_EPOCHS = int(args.N_EPOCHS)

    # Construct a tf.data.Dataset
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: x / 255.)
    ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Build Flow
    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = [7, 7, 16]  # (H//4, W//4, C*16)
    bijector_cls = flow_tfk_layers.RealNVPBijector2_tfk
    bijector_args = {'input_shape': data_shape, 'shift_and_log_scale_layer': flow_tfk_layers.ShiftAndLogScaleResNet_tfk,
                     'n_filters_base': 32}
    flow = flow_tfk_models.Flow(
        bijector_cls, data_shape, base_distr_shape, **bijector_args)

    sample = tf.random.normal([1] + data_shape)
    flow(sample)
    print(flow.sample(1).shape)

    utils.print_summary(flow)

    # Custom Training Loop

    # loss function
    loss_fn = flow_tfk_models.get_loss_function(flow)

    @tf.function  # Adding the tf.function makes it about 10 times faster!!!
    def train_step(x):
        with tf.GradientTape() as tape:
            tape.watch(flow.trainable_variables)
            loss = loss_fn(x)
            gradients = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(zip(gradients, flow.trainable_variables))
        return loss

    optimizer = tfk.optimizers.Adam()
    print("Start Training on {} epochs".format(N_EPOCHS))
    t0 = time.time()
    train_loss_results = []
    count = 0
    loss_history = []
    for epoch in range(N_EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()

        for batch in ds:
            loss = train_step(batch)
            epoch_loss_avg.update_state(loss)
            count += 1
            if count == 100:
                loss_history.append(loss)
                count = 0

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(
            epoch, epoch_loss_avg.result()))

    t1 = time.time()
    training_time = t1 - t0
    print("Training time: ", np.round(training_time, 2), ' seconds')

    np.save('loss_history', np.array(loss_history))
    print('loss history saved')

    samples = flow.sample(9)
    samples_np = samples.numpy()
    np.save('samples', samples_np)
    print("9 samples saved")

    log_file.close()


if __name__ == '__main__':
    main()
