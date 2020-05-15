import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class ShiftAndLogScaleConvNetGlow(tfk.layers.Layer):

    """
    Convolutional Neural Networks
    Return shift and log scale for Affine Coupling Layer

    input shape (list): (H, W, C) input dimension of the network
    n_filters (int): Number of filters in the hidden layers

    Architecture of the networks
    2 hidden layers with n_filters
    last layer has 2 * C filters: output dimension (batch_size, H, W, 2*C)
    splitting along the channel dimension to obtain log(s) and t
    """

    def __init__(self, input_shape, n_filters, data_format='channels_last'):
        super(ShiftAndLogScaleConvNetGlow, self).__init__()
        self.conv1 = tfk.layers.Conv2D(filters=n_filters, kernel_size=3,
                                       input_shape=input_shape, data_format=data_format,
                                       activation='relu', padding='same')
        self.conv2 = tfk.layers.Conv2D(
            filters=n_filters, kernel_size=1, activation='relu', padding='same')
        self.conv3 = tfk.layers.Conv2D(
            filters=2 * input_shape[-1], kernel_size=3, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        log_s, t = tf.split(x, num_or_size_splits=2, axis=-1)
        # !! Without the hyperbolic tangeant activation: Get positive log_prob !!
        log_s = tfk.activations.tanh(log_s)
        return log_s, t

class AffineCouplingLayerSplit(tfb.Bijector):

    """
    Affine Coupling Layer (Bijector):
    input should be 4 dimensionals: (batch_size, H, W, C)
    x1, x2 = split(x) # splitting along the channel dimension
    log(s), t = shift_and_log_scale_fn(x2)
    y2 = exp(log(s)) * x2 + t
    y1 = x1
    y = concat(y1, y2)
    """

    def __init__(self, shift_and_log_scale_fn, split_axis=-1):
        super(AffineCouplingLayerSplit, self).__init__(
            forward_min_event_ndims=3)
        self.shift_and_log_scale_fn = shift_and_log_scale_fn
        self.split_axis = split_axis

    def _forward(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.split_axis)
        log_s, t = self.shift_and_log_scale_fn(x2)
        y1 = x1
        y2 = tf.exp(log_s) * x2 + t
        y = tf.concat([y1, y2], axis=self.split_axis)
        return y

    def _inverse(self, y):
        y1, y2 = tf.split(y, 2, self.split_axis)
        log_s, t = self.shift_and_log_scale_fn(y2)
        x1 = y1
        x2 = (y2 - t) / tf.exp(log_s)
        x = tf.concat([x1, x2], axis=-1)
        return x

    def _inverse_log_det_jacobian(self, y):
        y1, y2 = tf.split(y, 2, self.split_axis)
        log_s, _ = self.shift_and_log_scale_fn(y2)
        log_det = log_s
        return -tf.reduce_sum(log_det, [1, 2, 3])


class AffineCouplingLayerMasked(tfb.Bijector):
    """
    Affine Coupling Layer (bijector) using binary masked
    as described in Real NVP

    Parameters:
        event_shape (list): dimension of the input data
        shift_anf_log_scale_fn: NN returning log scale and shift
        masking (str): either 'channel' or 'checkboard'

    Perform coupling layer with a binary masked b:
    forward:
        x = b * y + ((1-b) * y - t) * tf.exp(-log_s)
    inverse:
        y = b * x + (1-b) * (x * tf.exp(log_s) + t)
    log_det:
        Sum of tf.math.log(tf.abs(s))

    """

    def __init__(self, event_shape, shift_and_log_scale_fn, masking='channel', mask_state=0):
        super(AffineCouplingLayerMasked, self).__init__(
            forward_min_event_ndims=3)
        self.shift_and_log_scale_fn = shift_and_log_scale_fn
        self.binary_masked = self.binary_masked_fn(
            event_shape, masking, mask_state)

    @staticmethod
    def binary_masked_fn(input_shape, masking, mask_state):
        if masking == 'channel':
            assert(input_shape[-1] % 2 == 0)
            sub_shape = np.copy(input_shape)
            sub_shape[-1] = sub_shape[-1] // 2
            binary_masked = np.concatenate([np.ones(sub_shape),
                                            np.zeros(sub_shape)],
                                           axis=-1)
        if masking == 'checkboard':
            column_odd = [k % 2 for k in range(input_shape[-2])]
            column_even = [(k + 1) % 2 for k in range(input_shape[-2])]
            binary_masked = np.zeros((input_shape[-3], input_shape[-2]))
            for j in range(input_shape[-2]):
                if j % 2:
                    binary_masked[:, j] = column_even
                else:
                    binary_masked[:, j] = column_odd
            binary_masked = binary_masked.reshape(
                list(binary_masked.shape) + [1])
            binary_masked = np.repeat(binary_masked, input_shape[-1], axis=-1)

        binary_masked = binary_masked.reshape([1] + list(binary_masked.shape))
        if mask_state:
            return binary_masked
        else:
            return 1 - binary_masked

    def _forward(self, x):
        b = np.repeat(self.binary_masked, x.shape[0], axis=0)
        log_s, t = self.shift_and_log_scale_fn(x * b)
        y = b * x + (1 - b) * (x * tf.exp(log_s) + t)
        return y

    def _inverse(self, y):
        b = np.repeat(self.binary_masked, y.shape[0], axis=0)
        log_s, t = self.shift_and_log_scale_fn(y * b)
        x = b * y + (1 - b) * ((y - t) * tf.exp(-log_s))
        return x

    def _forward_log_det_jacobian(self, x):
        b = np.repeat(self.binary_masked, x.shape[0], axis=0)
        log_s, _ = self.shift_and_log_scale_fn(x * b)
        log_det = log_s * (1 - b)
        return tf.reduce_sum(log_det, axis=[1, 2, 3])

    def _inverse_log_det_jacobian(self, y):
        b = np.repeat(self.binary_masked, y.shape[0], axis=0)
        log_s, _ = self.shift_and_log_scale_fn(y * b)
        log_det = -log_s * (1 - b)
        return tf.reduce_sum(log_det, axis=[1, 2, 3])


class Squeeze(tfb.Reshape):
    """
    Squeezing operation as described in Real NVP
    """

    def __init__(self, event_shape_in):
        H, W, C = event_shape_in
        assert(H % 2 == 0)
        assert(W % 2 == 0)

        self.event_shape_out = (H // 2, W // 2, 4 * C)

        super(Squeeze, self).__init__(self.event_shape_out,
                                      event_shape_in)


class RealNVPStep(tfb.Bijector):
    """
    Real NVP Step: 3 affine coupling layers with alternate masking

    Parameter:
        event_shape
        shift_and_log_scale_layer:
            tf.keras.layers -> needs to be instantiate
    """

    def __init__(self, event_shape, shift_and_log_scale_layer, n_filters, masking):
        super(RealNVPStep, self).__init__(forward_min_event_ndims=3)

        self.shift_and_log_scale_1 = shift_and_log_scale_layer(
            event_shape, n_filters)
        self.coupling_layer_1 = AffineCouplingLayerMasked(
            event_shape, self.shift_and_log_scale_1, masking, 0)

        self.shift_and_log_scale_2 = shift_and_log_scale_layer(
            event_shape, n_filters)
        self.coupling_layer_2 = AffineCouplingLayerMasked(
            event_shape, self.shift_and_log_scale_2, masking, 1)

        self.shift_and_log_scale_3 = shift_and_log_scale_layer(
            event_shape, n_filters)
        self.coupling_layer_3 = AffineCouplingLayerMasked(
            event_shape, self.shift_and_log_scale_3, masking, 0)

        self.bijector = tfb.Chain(
            [self.coupling_layer_3, self.coupling_layer_2, self.coupling_layer_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

    def _inverse_log_det_jacobian(self, x):
        return self.bijector.inverse_log_det_jacobian(x, event_ndims=3)


class RealNVPBlock(tfb.Bijector):
    """
    Real NVP Block : 
    real nvp step (checkboard) -> squeezing -> real nvp step (channel)

    Parameter:
        event_shape
        shift_and_log_scale_layer:
            tf.keras.layers -> needs to be instantiate
    """

    def __init__(self, event_shape, shift_and_log_scale_layer, n_filters, batch_norm=False):
        super(RealNVPBlock, self).__init__(forward_min_event_ndims=3)

        self.coupling_step_1 = RealNVPStep(event_shape, shift_and_log_scale_layer,
                                           n_filters, 'checkboard')
        self.squeeze = Squeeze(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        self.coupling_step_2 = RealNVPStep(self.event_shape_out,
                                           shift_and_log_scale_layer, n_filters, 'channel')

        self.bijector = tfb.Chain(
            [self.coupling_step_2, self.squeeze, self.coupling_step_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

    def _inverse_log_det_jacobian(self, x):
        return self.bijector.inverse_log_det_jacobian(x, event_ndims=3)

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 2, W // 2, C * 4)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 2, W // 2, C * 4)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 2, W * 2, C // 4)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 2, W * 2, C // 4)


class RealNVPBijector(tfb.Bijector):
    def __init__(self, input_shape, shift_and_log_scale_layer, n_filters_base, batch_norm=False):
        super(RealNVPBijector, self).__init__(forward_min_event_ndims=3)

        self.real_nvp_block_1 = RealNVPBlock(input_shape, shift_and_log_scale_layer,
                                             n_filters_base, batch_norm)
        self.real_nvp_block_2 = RealNVPBlock(self.real_nvp_block_1.event_shape_out,
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, batch_norm)

        self.bijector = tfb.Chain(
            [self.real_nvp_block_2, self.real_nvp_block_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y):
        return self.bijector.inverse_log_det_jacobian(y, event_ndims=3)

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)
