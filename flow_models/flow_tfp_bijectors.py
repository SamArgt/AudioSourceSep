import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
tfd = tfp.distributions
tfb = tfp.bijectors
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
        x = self.batch_norm_1(x)
        x = self.activation_1(x)
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.activation_2(x)
        x = self.conv3(x)
        log_s, t = tf.split(x, num_or_size_splits=2, axis=-1)
        # !! Without the hyperbolic tangeant activation:
        # Get nan !!
        log_s = self.activation_log_s(log_s)
        return log_s, t


class AffineCouplingLayerMasked(tfb.Bijector):
    """
    Affine Coupling Layer (bijector) using binary masked
    as described in Real NVP

    Parameters:
        event_shape (list): dimension of the input data
        shift_anf_log_scale_layer: tfk.layers.Layer class
        n_hidden_units : number of hidden units in the shift_and_log_scale layer
            if shift_anf_log_scale_layer is a ConvNet: number of filters in the hidden layers
            if it is a DenseNet: number of units in the hidden layers

    Perform coupling layer with a binary masked b:
    forward:
        y = b * x + (1-b) * (x * tf.exp(log_s) + t)
    inverse:
        x = b * y + ((1-b) * y - t) * tf.exp(-log_s)
    log_det:
        Sum of log_s * (1 - b)

    """

    def __init__(self, event_shape, shift_and_log_scale_layer, n_hidden_units,
                 masking='channel', mask_state=0, dtype=tf.float32):
        super(AffineCouplingLayerMasked, self).__init__(
            forward_min_event_ndims=0)
        self.shift_and_log_scale_fn = shift_and_log_scale_layer(
            event_shape, n_hidden_units)
        self.binary_mask = self.binary_mask_fn(
            event_shape, masking, mask_state, dtype)
        self.tensor_dtype = dtype

    def _forward(self, x):
        b = tf.repeat(self.binary_mask, x.shape[0], axis=0)
        log_s, t = self.shift_and_log_scale_fn(x * b)
        y = b * x + (1 - b) * (x * tf.exp(log_s) + t)
        return y

    def _inverse(self, y):
        b = tf.repeat(self.binary_mask, y.shape[0], axis=0)
        log_s, t = self.shift_and_log_scale_fn(y * b)
        x = b * y + (1 - b) * ((y - t) * tf.exp(-log_s))
        return x

    def _forward_log_det_jacobian(self, x):
        b = tf.repeat(self.binary_mask, x.shape[0], axis=0)
        log_s, _ = self.shift_and_log_scale_fn(x * b)
        log_det = log_s * (1 - b)
        return log_det

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
        event_shape: list of int
        shift_and_log_scale_layer: tfk.layers.Layer class
        n_hidden_units : number of hidden units in the shift_and_log_scale layer
            if shift_anf_log_scale_layer is a ConvNet: number of filters in the hidden layers
            if it is a DenseNet: number of units in the hidden layers

    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_hidden_units, masking, dtype=tf.float32):
        super(RealNVPStep, self).__init__(forward_min_event_ndims=0)

        self.coupling_layer_1 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 0, dtype=dtype)

        self.coupling_layer_2 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 1, dtype=dtype)

        self.coupling_layer_3 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 0, dtype=dtype)

        self.bijector = tfb.Chain(
            [self.coupling_layer_3,
             self.coupling_layer_2,
             self.coupling_layer_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

    def _forward_log_det_jacobian(self, x):
        return self.bijector._forward_log_det_jacobian(x)


class RealNVPBlock(tfb.Bijector):
    """
    Real NVP Block :
    real nvp step (checkboard) -> squeezing -> real nvp step (channel)
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class

    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_filters, batch_norm=False):
        super(RealNVPBlock, self).__init__(forward_min_event_ndims=3)

        self.coupling_step_1 = RealNVPStep(event_shape,
                                           shift_and_log_scale_layer,
                                           n_filters, 'checkboard')
        self.squeeze = Squeeze(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        self.coupling_step_2 = RealNVPStep(self.event_shape_out,
                                           shift_and_log_scale_layer,
                                           n_filters, 'channel')

        self.bijector = tfb.Chain(
            [self.coupling_step_2, self.squeeze, self.coupling_step_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)

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

    """
    Real NVP Bijector :
    2 real NVP Blocks with a multiscale architecture as described in  the Real NVP paper
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class
    """

    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector, self).__init__(forward_min_event_ndims=3)

        self.real_nvp_block_1 = RealNVPBlock(input_shape,
                                             shift_and_log_scale_layer,
                                             n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock([H1, W1, C1 // 2],
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, batch_norm)

    def _forward(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        z2 = self.real_nvp_block_2.forward(h1)
        return tf.concat((z1, z2), axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.real_nvp_block_2.inverse(z2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1.inverse(output1)

    def _forward_log_det_jacobian(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        log_det_1 = self.real_nvp_block_1.forward_log_det_jacobian(
            x, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2.forward_log_det_jacobian(
            h1, event_ndims=3)
        return log_det_1 + log_det_2

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


class RealNVPBijector2(tfb.Bijector):
    """
    Real NVP Bijector :
    2 real NVP Blocks + additionnal Real NVP Step
        with a multiscale architecture as described in  the Real NVP paper
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class
    """

    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector2, self).__init__(forward_min_event_ndims=3)

        self.real_nvp_block_1 = RealNVPBlock(input_shape,
                                             shift_and_log_scale_layer,
                                             n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock([H1, W1, C1 // 2],
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, batch_norm)

        H2, W2, C2 = self.real_nvp_block_2.event_shape_out

        self.real_nvp_step = RealNVPStep([H2, W2, C2 // 2],
                                         shift_and_log_scale_layer,
                                         2 * n_filters_base, masking='checkboard')

    def _forward(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        output2 = self.real_nvp_block_2.forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        z3 = self.real_nvp_step.forward(h2)
        return tf.concat((z1, z2, z3), axis=-1)

    def _inverse(self, y):
        z1, z23 = tf.split(y, 2, axis=-1)
        z2, z3 = tf.split(z23, 2, axis=-1)
        h2 = self.real_nvp_step.inverse(z3)
        output2 = tf.concat((z2, h2), axis=-1)
        h1 = self.real_nvp_block_2.inverse(output2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1.inverse(output1)

    def _forward_log_det_jacobian(self, y):
        output1 = self.real_nvp_block_1.forward(y)
        log_det_1 = self.real_nvp_block_1.forward_log_det_jacobian(
            y, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2.forward_log_det_jacobian(
            h1, event_ndims=3)
        output2 = self.real_nvp_block_2.forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        log_det_3 = self.real_nvp_step.forward_log_det_jacobian(
            h2, event_ndims=3)
        return log_det_1 + log_det_2 + log_det_3

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


class ActNorm(tfb.Bijector):
    """
    Activation Normalization layer as described in the Glow paper
    The post_actnorm activations per channel have zero mean and unit variance
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameters:
        event_shape: [H, W, C]
        minibatch: [N, H, W, C]
    """

    def __init__(self, event_shape, minibatch, name='ActNorm'):
        super(ActNorm, self).__init__(forward_min_event_ndims=3)

        self.H, self.W, self.C = event_shape
        _, minibatch_H, minibatch_W, minibatch_C = minibatch.shape
        assert(self.H == minibatch_H)
        assert(self.W == minibatch_W)
        assert(self.C == minibatch_C)

        mean_init = tf.reduce_mean(minibatch, axis=[0, 1, 2])
        std_init = tf.math.reduce_std(minibatch, axis=[0, 1, 2]) + tf.constant(10**(-8), dtype=tf.float32)
        scale_init = 1 / std_init
        shift_init = - mean_init / std_init

        self.scale = tf.Variable(
            initial_value=scale_init, name=name + '/scale')
        self.shift = tf.Variable(
            initial_value=shift_init, name=name + '/shift')

    def _forward(self, x):
        return x * self.scale + self.shift

    def _inverse(self, y):
        return (y - self.shift) / (self.scale + tf.constant(10**(-8), dtype=tf.float32))

    def _forward_log_det_jacobian(self, x):
        log_det = self.H * self.W * \
            tf.reduce_sum(tf.math.log(tf.abs(self.scale)))
        return tf.repeat(log_det, x.shape[0], axis=0)


class Invertible1x1Conv(tfb.Bijector):

    """
    Invertible 1x1 Convolution as described in the Glow paper
    !! This bijector accepts only 3 dimensionals input !!

    Parameters:
        event_shape: [H, W, C]

    """

    def __init__(self, event_shape, name='inv1x1conv'):
        super(Invertible1x1Conv, self).__init__(forward_min_event_ndims=3)
        self.height, self.width, self.C = event_shape

        np_w = np.linalg.qr(np.random.randn(self.C, self.C))[0]
        self.w_shape = [self.C, self.C]

        np_p, np_l, np_u = scipy.linalg.lu(np_w)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(abs(np_s))
        np_u = np.triu(np_u, k=1)

        # Non Trainable Variable
        self.P = tf.Variable(initial_value=np_p, name=name + '/P', trainable=False, dtype=tf.float32)
        p_inv = tf.linalg.inv(self.P)
        self.P_inv = tf.Variable(initial_value=p_inv, name=name + '/P_inv', trainable=False, dtype=tf.float32)
        self.Sign_s = tf.Variable(
            name=name + "/sign_S", initial_value=np_sign_s, trainable=False, dtype=tf.float32)

        # Trainable Variables
        self.L = tf.Variable(name=name + "/L", initial_value=np_l, dtype=tf.float32)
        self.Log_s = tf.Variable(name=name + "/log_S", initial_value=np_log_s, dtype=tf.float32)
        self.U = tf.Variable(name=name + "/U", initial_value=np_u, dtype=tf.float32)

        # Triangular mask
        self.l_mask = np.tril(np.ones(self.w_shape, dtype=np.float32), -1)

    def _forward(self, x):
        L = self.L * self.l_mask + tf.eye(*self.w_shape, dtype=tf.float32)
        u = self.U * np.transpose(self.l_mask) + tf.linalg.diag(self.Sign_s * tf.exp(self.Log_s))
        w = tf.matmul(self.P, tf.matmul(L, u))
        w = tf.reshape(w, [1, 1, self.C, self.C])
        y = tf.nn.conv2d(x, filters=w, strides=[1, 1, 1, 1], padding='SAME')
        return y

    def _inverse(self, y):
        L = self.L * self.l_mask + tf.eye(*self.w_shape, dtype=tf.float32)
        u = self.U * np.transpose(self.l_mask) + tf.linalg.diag(self.Sign_s * tf.exp(self.Log_s))
        u_inv = tf.linalg.inv(u)
        l_inv = tf.linalg.inv(L)
        w_inv = tf.matmul(u_inv, tf.matmul(l_inv, self.P_inv))
        w_inv = tf.reshape(w_inv, [1, 1, self.C, self. C])
        x = tf.nn.conv2d(y, w_inv, [1, 1, 1, 1], 'SAME')
        return x

    def _forward_log_det_jacobian(self, x):
        log_det = self.height * self.width * \
            tf.reduce_sum(self.Log_s)
        return tf.repeat(log_det, x.shape[0], axis=0)


class GlowStep(tfb.Bijector):

    def __init__(self, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, name='glowStep'):

        super(GlowStep, self).__init__(forward_min_event_ndims=3)

        self.actnorm = ActNorm(event_shape, minibatch, name=name + '/ActNorm')
        self.inv1x1conv = Invertible1x1Conv(
            event_shape, name=name + '/inv1x1conv')
        self.coupling_layer = AffineCouplingLayerMasked(event_shape, shift_and_log_scale_layer,
                                                        n_hidden_units)
        self.bijector = tfb.Chain(
            [self.coupling_layer, self.inv1x1conv, self.actnorm])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)


class GlowBlock(tfb.Bijector):

    def __init__(self, K, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, name='glowBlock'):

        super(GlowBlock, self).__init__(forward_min_event_ndims=3)

        self.squeeze = Squeeze(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        minibatch_reshaped = self.squeeze.forward(minibatch)
        self.glow_steps = []
        for k in range(K):
            self.glow_steps.append(GlowStep(self.event_shape_out,
                                            shift_and_log_scale_layer,
                                            n_hidden_units, minibatch_reshaped,
                                            name=name + '/' + 'glowStep' + str(k)))

        self.chain = self.glow_steps + [self.squeeze]
        self.bijector = tfb.Chain(self.chain)

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)

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


class GlowBijector_2blocks(tfb.Bijector):

    def __init__(self, K, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch):

        super(GlowBijector_2blocks, self).__init__(forward_min_event_ndims=3)

        H, W, C = event_shape
        self.glow_block1 = GlowBlock(K, event_shape,
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch,
                                     name='glowBlock1')
        H1, W1, C1 = self.glow_block1.event_shape_out
        N, _, _, _ = minibatch.shape
        minibatch_reshape = tf.reshape(minibatch, [N] + [H1, W1, C1])
        _, minibatch_split = tf.split(minibatch_reshape, 2, axis=-1)

        self.glow_block2 = GlowBlock(K, [H1, W1, C1 // 2],
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch_split,
                                     name='glowBlock2')

    def _forward(self, x):
        output1 = self.glow_block1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        z2 = self.glow_block2.forward(h1)
        return tf.concat((z1, z2), axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.glow_block2.inverse(z2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.glow_block1.inverse(output1)

    def _forward_log_det_jacobian(self, x):
        output1 = self.glow_block1.forward(x)
        log_det_1 = self.glow_block1.forward_log_det_jacobian(
            x, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.glow_block2.forward_log_det_jacobian(
            h1, event_ndims=3)
        return log_det_1 + log_det_2

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
