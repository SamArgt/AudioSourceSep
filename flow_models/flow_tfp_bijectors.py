import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


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
                 masking='channel', mask_state=0, name='AffineCouplingLayer', dtype=tf.float32):
        super(AffineCouplingLayerMasked, self).__init__(
            forward_min_event_ndims=0)
        self.shift_and_log_scale_fn = shift_and_log_scale_layer(
            event_shape, n_hidden_units, name=name + '/shiftAndLogScaleLayer')
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


class AffineCouplingLayerSplit(tfb.Bijector):
    def __init__(self, event_shape, shift_and_log_scale_layer, n_hidden_units, name='AffineCouplingLayer'):
        super(AffineCouplingLayerSplit, self).__init__(forward_min_event_ndims=3)

        self.H, self.W, self.C = event_shape
        assert(self.C % 2 == 0)

        self.shift_and_log_scale_fn = shift_and_log_scale_layer(
            [self.H, self.W, self.C // 2], n_hidden_units, name=name + '/shiftAndLogScaleLayer')

    def _forward(self, x):
        xa, xb = tf.split(x, 2, axis=-1)
        log_s, t = self.shift_and_log_scale_fn(xb)
        # s = tf.nn.sigmoid(log_s)
        s = tf.exp(log_s)
        ya = s * xa + t
        yb = xb
        return tf.concat([ya, yb], axis=-1)

    def _inverse(self, y):
        ya, yb = tf.split(y, 2, axis=-1)
        log_s, t = self.shift_and_log_scale_fn(yb)
        # s = tf.nn.sigmoid(log_s)
        s = tf.exp(log_s)
        xa = (ya - t) / s
        xb = yb
        return tf.concat([xa, xb], axis=-1)

    def _forward_log_det_jacobian(self, x):
        xa, xb = tf.split(x, 2, axis=-1)
        log_s, _ = self.shift_and_log_scale_fn(xb)
        # s = tf.nn.sigmoid(log_s)
        # s = tf.exp(log_s)
        return tf.reduce_sum(log_s, axis=[1, 2, 3])


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
        log_scale_init = tf.math.log(scale_init)
        shift_init = - mean_init / std_init

        self.log_scale = tf.Variable(initial_value=log_scale_init, name=name + '/log_scale')
        # self.scale = tf.Variable(initial_value=scale_init, name=name + '/scale')
        self.shift = tf.Variable(
            initial_value=shift_init, name=name + '/shift')

    def _forward(self, x):
        return x * tf.exp(self.log_scale) + self.shift
        # return x * self.scale + self.shift

    def _inverse(self, y):
        return (y - self.shift) / tf.exp(self.log_scale)
        # return (y - self.shift) / self.scale

    def _forward_log_det_jacobian(self, x):
        log_det = self.H * self.W * tf.reduce_sum(self.log_scale)
        # log_det = self.H * self.W * tf.reduce_sum(tf.math.log(tf.abs(self.scale)))
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


class Preprocessing(tfp.bijectors.Bijector):
    def __init__(self, event_shape, alpha=0.05):
        super(Preprocessing, self).__init__(forward_min_event_ndims=3)
        self.alpha = alpha
        self.H, self.W, self.C = event_shape

    def _forward(self, x):
        x = self.alpha + (1 - self.alpha) * x
        return tf.math.log(x / (1 - x))

    def _inverse(self, y):
        y = 1 / (tf.exp(-y) + 1)
        return (y - self.alpha) / (1 - self.alpha)

    def _forward_log_det_jacobian(self, x):
        u = self.alpha + (1 - self.alpha) * x
        log_det = tf.math.log((1 - self.alpha)) - tf.math.log(u) - tf.math.log(1 - u)
        log_det = tf.reduce_sum(log_det, axis=[1, 2, 3])
        return log_det
