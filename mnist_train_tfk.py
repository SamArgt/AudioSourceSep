import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flow_models import layers
from flow_models import flow_tfk_models
from flow_models import utils
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

bijector_cls = layers.RealNVPStep_tfk
bijector_args = {'event_shape': data_shape, 'shift_and_log_scale_layer': layers.ShiftAndLogScaleResNet,
                 'n_filters': 32, 'masking': 'checkboard'}
flow = flow_tfk_models.Flow(bijector_cls, data_shape, **bijector_args)

sample = tf.random.normal([1] + data_shape)
flow(sample)

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


optimizer = tf.keras.optimizers.Adam()
NUM_EPOCHS = 100
print("Start Training on {} epochs".format(NUM_EPOCHS))
train_loss_results = []
for epoch in range(NUM_EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for batch in ds.as_numpy_iterator():
        loss = train_step(batch)
        if loss < 0:
            print("NEGATIVE LOSS")
            print(loss)
            break
        epoch_loss_avg.update_state(loss)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())

    print("Epoch {:03d}: Loss: {:.3f}".format(
        epoch, epoch_loss_avg.result()))

to_save = input('Save sample: Y/N ')
if to_save == 'Y':
    samples = flow.sample(10)
    samples_np = samples.numpy()
    np.save('samples', samples_np)
