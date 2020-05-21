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
tfd = tfp.distributions
tfb = tfp.bijectors
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
    ds = ds.shuffle(1024).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]

    # Build Flow
    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)
    K = 32

    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet
    n_filters_base = 128

    bijector = flow_glow.GlowBijector_2blocks(
        K, data_shape, shift_and_log_scale_layer, n_filters_base, minibatch)
    inv_bijector = tfb.Invert(bijector)

    print('Glow Bijector 2 Blocks: K = {} ; ShiftAndLogScaleResNet; n_filters = {}'.format(
        K, n_filters_base))

    flow = tfd.TransformedDistribution(tfd.Normal(
        0., 1.), inv_bijector, event_shape=base_distr_shape)
    print("flow sample shape: ", flow.sample(1).shape)
    utils.print_summary(flow)

    ckpt = tf.train.Checkpoint(variables=flow.variables)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    # Custom Training Loop

    @tf.function  # Adding the tf.function makes it about 10 times faster!!!
    def train_step(X):
        with tf.GradientTape() as tape:
            tape.watch(flow.trainable_variables)
            loss = -tf.reduce_mean(flow.log_prob(X))
            gradients = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(zip(gradients, flow.trainable_variables))
        return loss

    optimizer = tf.keras.optimizers.Adam()
    print("Start  Training on {} epochs".format(N_EPOCHS))
    t0 = time.time()
    train_loss_results = []
    loss_history = []
    count = 0
    is_nan_loss = False
    for epoch in range(N_EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        if is_nan_loss:
            break
        for batch in ds:
            loss = train_step(batch)
            epoch_loss_avg.update_state(loss)
            count += 1
            if count == 200:
                if tf.math.is_nan(loss):
                    print('Nan Loss')
                    is_nan_loss = True
                    break
                loss_history.append(loss)
                count = 0

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        if epoch % 10 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))
            manager.save()

    manager.save()
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
