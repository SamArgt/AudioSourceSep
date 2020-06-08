import tensorflow as tf
import numpy as np
tfk = tf.keras


class ShiftAndLofScaleDenseNet(tfk.layers.Layer):

    def __init__(self, input_shape, units):
        super(ShiftAndLofScaleDenseNet, self).__init__()

        self.dense1 = tfk.layers.Dense(
            units, activation='relu', input_shape=input_shape)
        self.dense2 = tfk.layers.Dense(units, activation='relu')
        self.dense3 = tfk.layers.Dense(units, activation='relu')
        self.dense4 = tfk.layers.Dense(units, activation='relu')
        self.dense5 = tfk.layers.Dense(input_shape[0] * 2, activation=None)
        self.activation_log_s = tfk.layers.Activation('tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        log_s, t = tf.split(x, 2, axis=-1)
        log_s = self.activation_log_s(log_s)
        return log_s, t


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

    def __init__(self, input_shape, n_filters, data_format='channels_last', name='ShiftAndLogScaleResNet', dtype=tf.float32):
        super(ShiftAndLogScaleResNet, self).__init__(dtype=dtype, name=name)
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
            filters=2 * input_shape[-1], kernel_size=3, padding='same', dtype=dtype,
            kernel_initializer='zeros', bias_initializer='zeros')
        self.activation_log_s = tfk.layers.Activation('tanh', dtype=dtype)

    def call(self, inputs):
        # if dtype = tf.float64, batch norm layers return an error
        x = self.conv1(inputs)
        x = self.batch_norm_1(x)
        x = self.activation_1(x)
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.activation_2(x)
        x = self.conv3(x)
        log_s, t = tf.split(x, num_or_size_splits=2, axis=-1)
        # !! Without the hyperbolic tangeant activation:
        # Get nan !!
        # log_s = self.activation_log_s(log_s)
        return log_s, t


class AffineCouplingLayerMasked_tfk(tfk.layers.Layer):
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
        super(AffineCouplingLayerMasked_tfk, self).__init__()
        self.shift_and_log_scale_fn = shift_and_log_scale_layer(
            event_shape, n_filters, dtype=dtype)
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
            assert(len(input_shape) == 3)
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


class RealNVPStep_tfk(tfk.layers.Layer):
    """
    Real NVP Step: 3 affine coupling layers with alternate masking

    Parameter:
        event_shape
        shift_and_log_scale_layer:
            tf.keras.layers -> needs to be instantiate
    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_filters, masking, dtype=tf.float32):
        super(RealNVPStep_tfk, self).__init__()

        self.coupling_layer_1 = AffineCouplingLayerMasked_tfk(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 0, dtype=dtype)

        self.coupling_layer_2 = AffineCouplingLayerMasked_tfk(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 1, dtype=dtype)

        self.coupling_layer_3 = AffineCouplingLayerMasked_tfk(
            event_shape, shift_and_log_scale_layer, n_filters, masking, 0, dtype=dtype)

    def _forward(self, x):
        output1 = self._forward.coupling_layer_1._forward(x)
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


class Squeeze_tfk(tfk.layers.Layer):
    """
    Squeezing operation as described in Real NVP
    """

    def __init__(self, event_shape_in):
        H, W, C = event_shape_in
        assert(H % 2 == 0)
        assert(W % 2 == 0)
        self.event_shape_in = list(event_shape_in)
        self.event_shape_out = [H // 2, W // 2, 4 * C]

        super(Squeeze_tfk, self).__init__()

    def _forward(self, x):
        return tf.reshape(x, [x.shape[0]] + self.event_shape_out)

    def _inverse(self, x):
        return tf.reshape(x, [x.shape[0]] + self.event_shape_in)

    def call(self, x):
        return tf.reshape(x, [x.shape[0]] + self.event_shape_out)


class RealNVPBlock_tfk(tfk.layers.Layer):
    """
    Real NVP Block :
    real nvp step (checkboard) -> squeezing -> real nvp step (channel)

    Parameter:
        event_shape
        shift_and_log_scale_layer:
            tf.keras.layers -> needs to be instantiate
    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_filters, batch_norm=False):
        super(RealNVPBlock_tfk, self).__init__()

        self.coupling_step_1 = RealNVPStep_tfk(event_shape,
                                               shift_and_log_scale_layer,
                                               n_filters, 'checkboard')
        self.squeeze = Squeeze_tfk(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        self.coupling_step_2 = RealNVPStep_tfk(self.event_shape_out,
                                               shift_and_log_scale_layer,
                                               n_filters, 'channel')

    def _forward(self, x):
        output1 = self.coupling_step_1._forward(x)
        output1_reshape = self.squeeze._forward(output1)
        return self.coupling_step_2._forward(output1_reshape)

    def _inverse(self, y):
        output1_reshape = self.coupling_step_2._inverse(y)
        output1 = self.squeeze._inverse(output1_reshape)
        return self.coupling_step_1._inverse(output1)

    def _forward_log_det_jacobian(self, x):
        output1 = self.coupling_step_1._forward(x)
        log_det_1 = self.coupling_step_1._forward_log_det_jacobian(x)
        output1_reshape = self.squeeze._forward(output1)
        log_det_2 = self.coupling_step_2._forward_log_det_jacobian(
            output1_reshape)
        return log_det_1 + log_det_2

    def call(self, x):
        x = self.coupling_step_1(x)
        x = self.squeeze(x)
        x = self.coupling_step_2(x)
        return x


class RealNVPBijector_tfk(tfk.layers.Layer):
    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector_tfk, self).__init__()

        self.real_nvp_block_1 = RealNVPBlock_tfk(input_shape,
                                                 shift_and_log_scale_layer,
                                                 n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock_tfk([H1, W1, C1 // 2],
                                                 shift_and_log_scale_layer,
                                                 2 * n_filters_base, batch_norm)

    def _forward(self, x):
        output1 = self.real_nvp_block_1._forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, C * 4))
        z2 = self.real_nvp_block_2._forward(h1)
        return tf.concat((z1, z2), axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.real_nvp_block_2._inverse(z2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1._inverse(output1)

    def _forward_log_det_jacobian(self, y):
        output1 = self.real_nvp_block_1._forward(y)
        log_det_1 = self.real_nvp_block_1._forward_log_det_jacobian(y)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2._forward_log_det_jacobian(h1)
        return log_det_1 + log_det_2

    def call(self, x):
        output1 = self.real_nvp_block_1(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, C * 4))
        z2 = self.real_nvp_block_2(h1)
        return tf.concat((z1, z2), axis=-1)


class RealNVPBijector2_tfk(tfk.layers.Layer):
    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector2_tfk, self).__init__()

        self.real_nvp_block_1 = RealNVPBlock_tfk(input_shape,
                                                 shift_and_log_scale_layer,
                                                 n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock_tfk([H1, W1, C1 // 2],
                                                 shift_and_log_scale_layer,
                                                 2 * n_filters_base, batch_norm)

        H2, W2, C2 = self.real_nvp_block_2.event_shape_out

        self.real_nvp_step = RealNVPStep_tfk([H2, W2, C2 // 2],
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, masking='checkboard')

    def _forward(self, x):
        output1 = self.real_nvp_block_1._forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        output2 = self.real_nvp_block_2._forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        z3 = self.real_nvp_step._forward(h2)
        return tf.concat((z1, z2, z3), axis=-1)

    def _inverse(self, y):
        z1, z23 = tf.split(y, 2, axis=-1)
        z2, z3 = tf.split(z23, 2, axis=-1)
        h2 = self.real_nvp_step._inverse(z3)
        output2 = tf.concat((z2, h2), axis=-1)
        h1 = self.real_nvp_block_2._inverse(output2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1._inverse(output1)

    def _forward_log_det_jacobian(self, y):
        output1 = self.real_nvp_block_1._forward(y)
        log_det_1 = self.real_nvp_block_1._forward_log_det_jacobian(
            y, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2._forward_log_det_jacobian(
            h1, event_ndims=3)
        output2 = self.real_nvp_block_2._forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        log_det_3 = self.real_nvp_step._forward_log_det_jacobian(
            h2, event_ndims=3)
        return log_det_1 + log_det_2 + log_det_3

    def call(self, x):
        output1 = self.real_nvp_block_1(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        output2 = self.real_nvp_block_2(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        z3 = self.real_nvp_step(h2)
        return tf.concat((z1, z2, z3), axis=-1)
