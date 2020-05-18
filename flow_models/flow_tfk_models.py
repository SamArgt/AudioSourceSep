import numpy as np
import tensorflow as tf
tfk = tf.keras


class Flow(tfk.Model):
    def __init__(self, bijector_cls, data_shape, base_distr_shape, **kwargs):
        super(Flow, self).__init__()
        self.event_shape = list(data_shape)
        self.base_distr_shape = list(base_distr_shape)
        self.bijector = bijector_cls(**kwargs)
        self.tensor_dtype = tf.float32

    def call(self, inputs):
        return self.bijector(inputs)

    def sample(self, num):
        z = tf.random.normal(
            shape=[num] + self.base_distr_shape, dtype=self.tensor_dtype)
        z = tf.cast(z, self.tensor_dtype)
        x = self.bijector._inverse(z)
        return x


def get_loss_function(flow):
    def loss_fn(x):
        z = flow(x)
        log_det = flow.losses
        log_pz = -0.5 * ((z**2) + tf.math.log(2 * tf.constant(np.pi)))
        log_pz = tf.reduce_sum(log_pz, axis=[1, 2, 3])
        log_prob = log_pz + log_det
        return -tf.reduce_mean(log_prob)

    return loss_fn
