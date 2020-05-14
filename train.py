import flow_models
import pipeline
import tensorflow as tf
import tensorflow_probability as tfp
import flow_models
import pipeline
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


# Prepraring Dataset
BATCH_SIZE = 16

dataset = pipeline.load_spec_tf("melSpecData")
spect_shape = list(dataset.take(1).as_numpy_iterator())[0].shape
print('shape of the spectrograms: ', spect_shape)
print(len(list(dataset.map(lambda x: 1).as_numpy_iterator())), ' spectrograms')
dataset = dataset.map(lambda x: tf.dtypes.cast(x, tf.float32))
dataset = dataset.map(lambda x: tf.reshape(x, list(spect_shape) + [1]))
dataset = dataset.batch(BATCH_SIZE)
print("shape of 1 element of the dataset: ",
      list(dataset.take(1).as_numpy_iterator())[0].shape)


# Preparing Flow Model
optimizer = tf.keras.optimizers.Adam()
num_epochs = 2

event_shape = [128, 431, 2]
n_filters = 32

model = flow_models.RealNVP(event_shape, n_filters, batch_norm=False)

# TRAINING


@tf.function  # Adding the tf.function makes it about 10 times faster!!!
def train_step(X):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(model.flow.log_prob(X))
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


train_loss_results = []
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for batch in dataset.as_numpy_iterator():
        loss = train_step(batch)
        print("loss: ", loss)
        epoch_loss_avg.update_state(loss)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())

    print("Epoch {:03d}: Loss: {:.3f}".format(
        epoch, epoch_loss_avg.result()))
