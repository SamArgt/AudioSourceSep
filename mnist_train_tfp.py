import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from flow_models import flow_tfp_bijectors
from flow_models import utils
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train', shuffle_files=True)
# Build your input pipeline
ds = ds.map(lambda x: x['image'])
ds = ds.map(lambda x: tf.cast(x, tf.float32))
ds = ds.map(lambda x: x / 255.)
ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# Build Flow
data_shape = [28, 28, 1]  # (H, W, C)
base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)

shift_and_log_scale_layer = flow_tfp_bijectors.ShiftAndLogScaleResNet
n_filters_base = 32

bijector = flow_tfp_bijectors.RealNVPBijector(
    (28, 28, 1), shift_and_log_scale_layer, n_filters_base)
inv_bijector = tfb.Invert(bijector)

flow = tfd.TransformedDistribution(tfd.Normal(
    0., 1.), inv_bijector, event_shape=base_distr_shape)
print("flow sample shape: ", flow.sample(1).shape)
utils.print_summary(flow)

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
NUM_EPOCHS = 100
print("Start  Training on {} epochs".format(NUM_EPOCHS))
train_loss_results = []
for epoch in range(NUM_EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for batch in ds.as_numpy_iterator():
        loss = train_step(batch)
        epoch_loss_avg.update_state(loss)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())

    print("Epoch {:03d}: Loss: {:.3f}".format(
        epoch, epoch_loss_avg.result()))

samples = flow.sample(9)
samples_np = samples.numpy()
np.save('samples_realnvp', samples_np)
