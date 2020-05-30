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
        description='Train Flow model on MNIST dataset')
    parser.add_argument('OUTPUT', type=str,
                        help='output dirpath for savings')
    parser.add_argument('--n_epochs', type=str, default=None,
                        help='number of epochs to train')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    args = parser.parse_args()
    output_dirpath = args.OUTPUT
    if args.restore is not None:
        restore_abs_dirpath = os.path.abspath(args.restore)

    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)

    log_file = open('out.log', 'w')
    sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Tensorboard
    # Clear any logs from previous runs
    try:
        shutil.rmtree('tensorboard_logs')
    except FileNotFoundError:
        pass
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(
        'tensorboard_logs', 'gradient_tape', current_time, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    tfk.backend.clear_session()

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Construct a tf.data.Dataset
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: x / 255.)
    batch_size = 256
    ds = ds.shuffle(1024).batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    ds = mirrored_strategy.experimental_distribute_dataset(ds)

    # Set flow parameters
    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)
    K = 32
    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet
    n_filters_base = 64

    # Build Flow
    with mirrored_strategy.scope():
        bijector = flow_glow.GlowBijector_2blocks(
            K, data_shape, shift_and_log_scale_layer, n_filters_base, minibatch)
        inv_bijector = tfb.Invert(bijector)
        flow = tfd.TransformedDistribution(tfd.Normal(
            0., 1.), inv_bijector, event_shape=base_distr_shape)

    params_str = 'Glow Bijector 2 Blocks: \n\t K = {} \n\t ShiftAndLogScaleResNet \n\t n_filters = {} \n\t batch size = {}'.format(
        K, n_filters_base, batch_size)
    print(params_str)
    with train_summary_writer.as_default():
        tf.summary.text(name='Flow parameters',
                        data=tf.constant(params_str), step=0)
    print("flow sample shape: ", flow.sample(1).shape)
    # utils.print_summary(flow)
    print("Total Trainable Variables: ", utils.total_trainable_variables(flow))

    # Custom Training Step
    # Adding the tf.function makes it about 10 times faster!!!
    @tf.function
    def train_step(dist_inputs):
        def step_fn(X):
            with tf.GradientTape() as tape:
                tape.watch(flow.trainable_variables)
                log_prob_sum = -tf.reduce_sum(flow.log_prob(X))
                scaled_loss = log_prob_sum * (1.0 / batch_size)
            gradients = tape.gradient(scaled_loss, flow.trainable_variables)
            optimizer.apply_gradients(list(zip(gradients, flow.trainable_variables)))
            return log_prob_sum
        per_example_losses = mirrored_strategy.run(
            step_fn, args=(dist_inputs,))
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                             per_example_losses, axis=0)
        return mean_loss

    # Training Parameters
    N_EPOCHS = 100
    if args.n_epochs is not None:
        N_EPOCHS = int(args.n_epochs)
    with mirrored_strategy.scope():
        optimizer = tf.keras.optimizers.Adam()

    print("Start Training on {} epochs".format(N_EPOCHS))

    # Checkpoint object
    ckpt = tf.train.Checkpoint(variables=flow.variables, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)
    # Restore weights if specified
    if args.restore is not None:
        ckpt.restore(tf.train.latest_checkpoint(
            os.path.join(restore_abs_dirpath, 'tf_ckpts')))

    t0 = time.time()
    loss_history = []
    history_loss_avg = tf.keras.metrics.Mean()
    epoch_loss_avg = tf.keras.metrics.Mean()
    count_step = optimizer.iterations.numpy()
    min_avg_loss, curr_avg_loss = 0., 0.
    loss_per_epoch = 10  # number of losses per epoch to save
    is_nan_loss = False
    # Custom Training Loop
    with mirrored_strategy.scope():
        for epoch in range(N_EPOCHS):
            epoch_loss_avg.reset_states()

            if is_nan_loss:
                break

            for batch in ds:
                loss = train_step(batch)
                epoch_loss_avg.update_state(loss)
                history_loss_avg.update_state(loss)
                count_step += 1

                if count_step % (60000 // (batch_size * loss_per_epoch)) == 0:

                    if tf.math.is_nan(loss):
                        print('Nan Loss')
                        is_nan_loss = True
                        break

                    loss_history.append(history_loss_avg.result())
                    with train_summary_writer.as_default():
                        step_int = int(loss_per_epoch * count_step * batch_size / 60000)
                        tf.summary.scalar(
                            'loss', history_loss_avg.result(), step=step_int)

                    history_loss_avg.reset_states()

            if (N_EPOCHS < 100) or (epoch % (N_EPOCHS // 100) == 0):
                print("Epoch {:03d}: Loss: {:.3f}".format(
                    epoch, epoch_loss_avg.result()))

                samples = flow.sample(9)
                samples = samples.numpy().reshape((9, 28, 28, 1))
                with train_summary_writer.as_default():
                    tf.summary.image("9 generated samples",
                                     samples, max_outputs=27, step=epoch)

                curr_avg_loss = epoch_loss_avg.result()
                if curr_avg_loss < min_avg_loss:
                    save_path = manager.save()
                    print("Model Saved at {}".format(save_path))
                    min_avg_loss = curr_avg_loss

    # Saving the last variables
    manager.save()
    print("Model Saved at {}".format(save_path))

    # Training Time
    t1 = time.time()
    training_time = t1 - t0
    print("Training time: ", np.round(training_time, 2), ' seconds')

    # Saving loss history
    np.save('loss_history', np.array(loss_history))
    print('loss history saved')

    # Saving samples
    samples = flow.sample(9)
    samples_np = samples.numpy()
    np.save('samples', samples_np)
    print("9 samples saved")

    log_file.close()


if __name__ == '__main__':
    main()