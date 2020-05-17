import tensorflow as tf
import numpy as np
tfk = tf.keras


class ShiftAndLogScaleResNet(tfk.layers.Layer):

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

    def __init__(self, input_shape, n_filters, data_format='channels_last', dtype=tf.float32):
        super(ShiftAndLogScaleResNet, self).__init__(dtype=dtype)
        self.conv1 = tfk.layers.Conv2D(filters=n_filters, kernel_size=3,
                                       input_shape=input_shape,
                                       data_format=data_format,
                                       activation='relu', padding='same', dtype=dtype)
        self.batch_norm_1 = tfk.layers.BatchNormalization(dtype=dtype)
        self.activation_1 = tfk.layers.Activation('relu', dtype=dtype)

        self.conv2 = tfk.layers.Conv2D(
            filters=n_filters, kernel_size=1, activation='relu',
            padding='same', dtype=dtype)
        self.batch_norm_2 = tfk.layers.BatchNormalization(dtype=dtype)
        self.activation_2 = tfk.layers.Activation('relu', dtype=dtype)

        self.conv3 = tfk.layers.Conv2D(
            filters=2 * input_shape[-1], kernel_size=3, padding='same', dtype=dtype)
        self.activation_log_s = tfk.layers.Activation('tanh', dtype=dtype)

    def call(self, inputs):
        # if dtype = tf.float64, batch norm layers return an error
        x = self.conv1(inputs)
        #x = self.batch_norm_1(x)
        x = self.activation_1(x)
        x = self.conv2(x)
        #x = self.batch_norm_2(x)
        x = self.activation_2(x)
        x = self.conv3(x)
        log_s, t = tf.split(x, num_or_size_splits=2, axis=-1)
        # !! Without the hyperbolic tangeant activation:
        # Get positive log_prob !!
        # For the AffineCouplingLayer at least...
        log_s = self.activation_log_s(log_s)
        return log_s, t


class AffineCouplingLayerMasked(tfk.layers.Layer):
    """
    Affine Coupling Layer (bijector) using binary masked
    as described in Real NVP

    Parameters:
        event_shape (list): dimension of the input data
        shift_anf_log_scale_fn: Layer class returning log scale and shift
        masking (str): either 'channel' or 'checkboard'

    Perform coupling layer with a binary masked b:
    forward:
        y = b * x + (1-b) * (x * tf.exp(log_s) + t)
        
    inverse:
        x = b * y + ((1-b) * (y - t)) * tf.exp(-log_s)
    log_det:
        Sum of log_s * (1-b)

    """

    def __init__(self, event_shape, shift_and_log_scale_layer, n_filters,
                 masking='channel', mask_state=0, dtype=tf.float32):
        super(AffineCouplingLayerMasked, self).__init__()
        self.shift_and_log_scale_fn = shift_and_log_scale_layer(event_shape, n_filters, dtype=dtype)
        self.binary_mask = tf.Variable(self.binary_mask_fn(
            event_shape, masking, mask_state, dtype), trainable=False, dtype=dtype)
        self.tensor_dtype = dtype

    def _forward(self, x):
        b = tf.repeat(self.binary_mask, x.shape[0], axis=0)
        x1 = x * b
        x2 = x * (1 - b)
        log_s, t = self.shift_and_log_scale_fn(x1)
        log_s, t = (1 - b) * log_s, (1 - b) * t
        y1 = x1
        y2 = x2 * tf.exp(log_s) + t
        return y1 + y2

    def _inverse(self, y):
        b = tf.repeat(self.binary_mask, y.shape[0], axis=0)
        y1 = y * b
        y2 = y * (1 - b)
        log_s, t = self.shift_and_log_scale_fn(y1)
        log_s, t = (1 - b) * log_s, (1 - b) * t
        x1 = y1
        x2 = (y2 - t) * tf.exp(-log_s)
        return x1 + x2

    def _forward_log_det_jacobian(self, x):
        b = tf.repeat(self.binary_mask, x.shape[0], axis=0)
        x1 = b * x
        log_s, _ = self.shift_and_log_scale_fn(x1)
        log_det = log_s * (1 - b)
        return tf.reduce_sum(log_det, axis=[1, 2, 3])

    def call(self, x):
        z = self._forward(x)
        log_det = self._forward_log_det_jacobian(x)
        self.add_loss(log_det)
        return z

    @staticmethod
    def binary_mask_fn(input_shape, masking, mask_state, dtype):
        if masking == 'channel':
            assert(input_shape[-1] % 2 == 0)
            sub_shape = np.copy(input_shape)
            sub_shape[-1] = sub_shape[-1] // 2
            binary_mask = np.concatenate([np.ones(sub_shape),
                                          np.zeros(sub_shape)],
                                         axis=-1)
        if masking == 'checkboard':
            column_odd = [k % 2 for k in range(input_shape[-2])]
            column_even = [(k + 1) % 2 for k in range(input_shape[-2])]
            binary_mask = np.zeros((input_shape[-3], input_shape[-2]))
            for j in range(input_shape[-2]):
                if j % 2:
                    binary_mask[:, j] = column_even
                else:
                    binary_mask[:, j] = column_odd
            binary_mask = binary_mask.reshape(
                list(binary_mask.shape) + [1])
            binary_mask = np.repeat(binary_mask, input_shape[-1], axis=-1)

        binary_mask = binary_mask.reshape([1] + list(binary_mask.shape))
        if mask_state:
            return tf.cast(binary_mask, dtype)
        else:
            return tf.cast((1 - binary_mask), dtype)


class RealNVPStep(tfk.layers.Layer):
    """
    Real NVP Step: 3 affine coupling layers with alternate masking

    Parameter:
        event_shape
        shift_and_log_scale_layer:
            tf.keras.layers -> needs to be instantiate
    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_filters, masking, dtype=tf.float32):
        super(RealNVPStep, self).__init__()

        self.coupling_layer_1 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 0, dtype=dtype)

        self.coupling_layer_2 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 1, dtype=dtype)

        self.coupling_layer_3 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 0, dtype=dtype)

    def _forward(self, x):
        output1 = self._forward.coupling_layer_1._forwardward(x)
        output2 = self._forward.coupling_layer_2._forward(output1)
        return self._forward.coupling_layer_3._forward(output2)

    def _inverse(self, y):
        output2 = self.coupling_layer_3._inverse(y)
        output1 = self.coupling_layer_2._inverse(output2)
        return self.coupling_layer_1._inverse(output1)

    def _forward_log_det_jacobian(self, x):
        log_det_1 = self.coupling_layer_1._forward_log_det_jacobian(x)
        output1 = self.coupling_layer_1._forward(x)
        log_det_2 = self.coupling_layer_2._forward_log_det_jacobian(output1)
        output2 = self.coupling_layer_2._forward(output1)
        log_det_3 = self.coupling_layer_3._forward_log_det_jacobian(output2)
        return log_det_1 + log_det_2 + log_det_3

    def call(self, x):
        x = self.coupling_layer_1(x)
        x = self.coupling_layer_2(x)
        x = self.coupling_layer_3(x)
        return x
