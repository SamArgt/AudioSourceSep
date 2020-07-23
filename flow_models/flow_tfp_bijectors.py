import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from .flow_tfk_layers import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class AffineCouplingLayerMasked(tfb.Bijector):
    """
    Affine Coupling Layer (bijector) using binary masked
    as described in Real NVP

    Parameters:
        event_shape (list of int): dimension of the input data
        shift_and_log_scale_layer: tfk.layers.Layer for computing the scale and shift parameters
        **kwargs: parameters for shift_and_log_scale_layer

    Perform coupling layer with a binary masked b:
    forward:
        y = b * x + (1-b) * (x * tf.exp(log_s) + t)
    inverse:
        x = b * y + ((1-b) * y - t) * tf.exp(-log_s)
    log_det:
        Sum of log_s * (1 - b)
    """

    def __init__(self, event_shape, shift_and_log_scale_layer, masking='channel', mask_state=0, name='AffineCouplingLayer', **kwargs):

        super(AffineCouplingLayerMasked, self).__init__(forward_min_event_ndims=0, name=name)
        self.shift_and_log_scale_fn = shift_and_log_scale_layer(event_shape, **kwargs)
        self.binary_mask = self.binary_mask_fn(event_shape, masking, mask_state)

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
    def binary_mask_fn(input_shape, masking, mask_state):
        """
        Compute the binary mask

        input_shape: shape of the binary mask
        masking: type of masking
        mask_state: 0 or 1. parameters to alternate between b and (1. - b)

        Return binary mask
        """
        if masking == 'channel':
            assert(input_shape[-1] % 2 == 0)
            sub_shape = np.copy(input_shape)
            sub_shape[-1] = sub_shape[-1] // 2
            binary_mask = np.concatenate([np.ones(sub_shape),
                                          np.zeros(sub_shape)],
                                         axis=-1)
        if masking == 'checkerboard':
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
            return tf.cast(binary_mask, tf.float32)
        else:
            return tf.cast((1 - binary_mask), tf.float32)


class StackedMaskedCouplingLayers(tfb.Bijector):
    """
    Stack multiple coupling layers with BatchNormalization after each one.
    Parameters:
        event_shape (list of int)
        n_coupling_layers: number of coupling layers to stack
        shift_and_log_scale_layer: tf.keras.layers.Layer
        masking: 'checkerboard' or 'channel'
        **kwargs: parameters for the shift_and_log_scale_layer
    """

    def __init__(self, event_shape, n_coupling_layers, shift_and_log_scale_layer, masking, **kwargs):
        super(StackedMaskedCouplingLayers, self).__init__(forward_min_event_ndims=3)

        self.stack_layers = []
        for i in range(n_coupling_layers):
            self.stack_layers += [tfb.BatchNormalization(),
                                  AffineCouplingLayerMasked(event_shape, shift_and_log_scale_layer,
                                                            masking, mask_state=i % 2, **kwargs)]
        self.bijector = tfb.Chain(self.stack_layers)

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)


class AffineCouplingLayerSplit(tfb.Bijector):
    def __init__(self, event_shape, shift_and_log_scale_layer, name='AffineCouplingLayer', **kwargs):
        super(AffineCouplingLayerSplit, self).__init__(
            forward_min_event_ndims=3, name=name)

        self.H, self.W, self.C = event_shape
        assert(self.C % 2 == 0)

        self.shift_and_log_scale_fn = shift_and_log_scale_layer([self.H, self.W, self.C // 2], **kwargs)

    def _forward(self, x):
        xa, xb = tf.split(x, 2, axis=-1)
        log_s, t = self.shift_and_log_scale_fn(xb)
        s = tf.exp(log_s)
        ya = s * xa + t
        yb = xb
        return tf.concat([ya, yb], axis=-1)

    def _inverse(self, y):
        ya, yb = tf.split(y, 2, axis=-1)
        log_s, t = self.shift_and_log_scale_fn(yb)
        s = tf.exp(log_s)
        xa = (ya - t) / s
        xb = yb
        return tf.concat([xa, xb], axis=-1)

    def _forward_log_det_jacobian(self, x):
        xa, xb = tf.split(x, 2, axis=-1)
        log_s, _ = self.shift_and_log_scale_fn(xb)
        return tf.reduce_sum(log_s, axis=[1, 2, 3])


class Squeeze(tfb.Bijector):
    """
    Squeezing operation as described in Real NVP
    """

    def __init__(self, event_shape_in, name="Squeeze"):
        super(Squeeze, self).__init__(forward_min_event_ndims=3, name=name)
        H, W, C = event_shape_in
        self.H, self.W, self.C = H, W, C
        assert(H % 2 == 0)
        assert(W % 2 == 0)

        self.event_shape_out = (H // 2, W // 2, 4 * C)

    def _forward(self, x):
        x = tf.reshape(x, [-1, self.H // 2, 2, self.W // 2, 2, self.C])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(x, [-1, self.H // 2, self.W // 2, self.C * 4])
        return x

    def _inverse(self, y):
        y = tf.reshape(y, [-1, self.H // 2, self.W // 2, self.C, 2, 2])
        y = tf.transpose(y, [0, 1, 4, 2, 5, 3])
        y = tf.reshape(y, [-1, self.H, self.W, self.C])
        return y

    def _forward_log_det_jacobian(self, x):
        return tf.zeros(shape=x.shape[0])

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


class ActNorm(tfb.Bijector):
    """
    Activation Normalization layer as described in the Glow paper
    The post_actnorm activations per channel have zero mean and unit variance
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameters:
        event_shape: [H, W, C]
        minibatch: [N, H, W, C]
    """

    def __init__(self, event_shape, minibatch, normalize='channel', name='ActNorm'):
        super(ActNorm, self).__init__(forward_min_event_ndims=3, name=name)

        self.H, self.W, self.C = event_shape
        _, minibatch_H, minibatch_W, minibatch_C = minibatch.shape
        assert(self.H == minibatch_H)
        assert(self.W == minibatch_W)
        assert(self.C == minibatch_C)

        if normalize == 'channel':
            mean_init = tf.reduce_mean(minibatch, axis=[0, 1, 2])
            std_init = tf.math.reduce_std(
                minibatch, axis=[0, 1, 2]) + tf.constant(10**(-8), dtype=tf.float32)

        elif normalize == 'all':
            mean_init, var_init = tf.nn.moments(minibatch, axes=[0])
            std_init = tf.math.sqrt(var_init) + \
                tf.constant(10**(-8), dtype=tf.float32)

        scale_init = 1 / std_init
        log_scale_init = tf.math.log(scale_init)
        shift_init = - mean_init / std_init

        self.log_scale = tf.Variable(
            initial_value=log_scale_init, name='log_scale')
        # self.scale = tf.Variable(initial_value=scale_init, name=name + '/scale')
        self.shift = tf.Variable(
            initial_value=shift_init, name='shift')

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
        super(Invertible1x1Conv, self).__init__(forward_min_event_ndims=3, name=name)
        self.height, self.width, self.C = event_shape

        np_w = np.linalg.qr(np.random.randn(self.C, self.C))[0]
        self.w_shape = [self.C, self.C]

        np_p, np_l, np_u = scipy.linalg.lu(np_w)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(abs(np_s))
        np_u = np.triu(np_u, k=1)

        # Non Trainable Variable
        self.P = tf.Variable(initial_value=np_p, name='P', trainable=False, dtype=tf.float32)
        p_inv = tf.linalg.inv(self.P)
        self.P_inv = tf.Variable(
            initial_value=p_inv, name='P_inv', trainable=False, dtype=tf.float32)
        self.Sign_s = tf.Variable(
            name="sign_S", initial_value=np_sign_s, trainable=False, dtype=tf.float32)

        # Trainable Variables
        self.L = tf.Variable(
            name="L", initial_value=np_l, dtype=tf.float32)
        self.Log_s = tf.Variable(name="log_S",
                                 initial_value=np_log_s, dtype=tf.float32)
        self.U = tf.Variable(
            name="U", initial_value=np_u, dtype=tf.float32)

        # Triangular mask
        self.l_mask = np.tril(np.ones(self.w_shape, dtype=np.float32), -1)

    def _forward(self, x):
        L = self.L * self.l_mask + tf.eye(*self.w_shape, dtype=tf.float32)
        u = self.U * np.transpose(self.l_mask) + \
            tf.linalg.diag(self.Sign_s * tf.exp(self.Log_s))
        w = tf.matmul(self.P, tf.matmul(L, u))
        w = tf.reshape(w, [1, 1, self.C, self.C])
        y = tf.nn.conv2d(x, filters=w, strides=[1, 1, 1, 1], padding='SAME')
        return y

    def _inverse(self, y):
        L = self.L * self.l_mask + tf.eye(*self.w_shape, dtype=tf.float32)
        u = self.U * np.transpose(self.l_mask) + \
            tf.linalg.diag(self.Sign_s * tf.exp(self.Log_s))
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


class ImgPreprocessing(tfp.bijectors.Bijector):
    """
    Dequantize and Preprocess images by adding uniform and with the following operation:
    x = logit(alpha + (1 - alpha) * x / 256.)
    Parameters:
        event_shape (list of int)
        alpha (float)
    """

    def __init__(self, event_shape, alpha=0.05, use_logit=True, name="ImgPreprocessing"):
        super(ImgPreprocessing, self).__init__(forward_min_event_ndims=3, name=name)
        self.alpha = alpha
        self.use_logit = use_logit
        self.event_shape = event_shape
        self.H, self.W, self.C = event_shape

    def _forward(self, x):
        x += tf.random.uniform(x.shape, minval=0., maxval=1.)
        if use_logit:
            x = self.alpha + (1. - 2 * self.alpha) * x / 256.
            x = tf.math.log(x) - tf.math.log(1. - x)
        else:
            x = x / 256. - 0.5
        return x

    def _inverse(self, y):
        if self.use_logit:
            y = tf.math.sigmoid(y)
            y = (y - self.alpha) * 256. / (1 - 2 * self.alpha)
        else:
            y = (y + 0.5) * 256.
        return y

    def _forward_log_det_jacobian(self, x):
        x += tf.random.uniform(x.shape, minval=0., maxval=1.)
        if self.use_logit:
            x = self.alpha + (1. - 2 * self.alpha) * x / 256.
            log_det = tf.math.log(x) - tf.math.log(1. - x) + tf.math.log((1. - 2 * self.alpha) / 256.)
        else:
            log_det = tf.ones(x.shape) / 256.
        log_det = tf.reduce_sum(log_det, axis=[1, 2, 3])
        return log_det


class SpecPreprocessing(tfp.bijectors.Bijector):
    def __init__(self, minval, maxval, alpha=1e-10, use_logit=True, name="SpecPreprocessing"):
        super(SpecPreprocessing, self).__init__(forward_min_event_ndims=3, name=name)
        self.maxval = maxval
        self.minval = minval
        self.alpha = alpha
        self.use_logit = use_logit

    def _forward(self, x):
        x = (x - self.minval) / self.maxval
        if self.use_logit:
            x = (1. - 2 * alpha) * x + alpha
            x = tf.math.log(x) - tf.math.log(1. - x)
        return x

    def _inverse(self, y):
        if self.use_logit:
            y = tf.math.sigmoid(y)
            y = (y - alpha) / (1. - 2 * alpha)
        y = y * self.maxval + self.minval
        return y

    def _forward_log_det_jacobian(self, x):
        x = (x - self.minval) / self.maxval
        if self.use_logit:
            x = (1. - 2 * alpha) * x + alpha
            log_det = -tf.math.log(x) - tf.math.log(1. - x) + tf.math.log((1. - 2 * alpha) / self.maxval)
        else:
            log_det = tf.ones(x.shape) / self.maxval
        return tf.reduce_sum(log_det, axis=[1, 2, 3])


class MixLogisticCDFAttnCoupling(tfp.bijectors.Bijector):

    """
    Mixture Logistic CDF Layer as described in Flow ++
    """

    def __init__(self, input_shape, split='channel', split_state=0, n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, context=False, name="MixLogCDFAttnCoupling"):

        super(MixLogisticCDFAttnCoupling, self).__init__(
            forward_min_event_ndims=3)
        self.H, self.W, self.C = input_shape
        self.split = split
        self.split_state = split_state
        if split == 'channel':
            assert self.C % 2 == 0
            nn_input_shape = [self.H, self.W, self.C // 2]
        elif split == 'checkerboard':
            assert self.W % 2 == 0
            nn_input_shape = [self.H, self.W // 2, self.C]
        else:
            raise ValueError('split should be channel or checkerboard')

        self.n_components = n_components
        self.nn = ConvAttnNet(input_shape=nn_input_shape, n_components=n_components,
                              n_blocks=n_blocks, filters=filters, dropout_p=dropout_p,
                              heads=heads, context=context, name=name + "/ConvAttnNet")

    def _forward(self, x, context=None):
        if self.split == 'channel':
            x1, x2 = tf.split(x, 2, axis=-1)
        else:
            x = tf.reshape(x, (-1, self.H, self.W // 2, 2, self.C))
            x1, x2 = tf.unstack(x, axis=3)

        if self.split_state:
            x2, x1 = x1, x2

        log_s, t, ml_logits, ml_means, ml_logscales = self.nn(x1, context=context)

        y1 = x1
        y2 = tf.exp(self.MixLog_logCDF(x2, ml_logits, ml_means, ml_logscales, self.n_components))
        y2 = tf.clip_by_value(y2, clip_value_min=float(1e-10), clip_value_max=float(1. - 1e-7))
        #assert tf.reduce_all(y2 < 1.), tf.reduce_max(y2)
        #assert tf.reduce_all(y2 > 0.), tf.reduce_min(y2)
        y2 = self.inv_sigmoid(y2)
        y2 = y2 * tf.exp(log_s) + t

        if self.split == 'channel':
            return tf.concat([y1, y2], axis=-1)
        else:
            y = tf.stack([y1, y2], axis=3)
            return tf.reshape(y, (-1, self.H, self.W, self.C))

    def _inverse(self, y, context=None):
        if self.split == "channel":
            y1, y2 = tf.split(y, 2, axis=-1)
        else:
            y = tf.reshape(y, (-1, self.H, self.W // 2, 2, self.C))
            y1, y2 = tf.unstack(y, axis=3)

        if self.split_state:
            y2, y1 = y1, y2

        log_s, t, ml_logits, ml_means, ml_logscales = self.nn(y1, context=context)

        x1 = y1
        x2 = (y2 - t) / tf.exp(log_s)
        x2 = tf.math.sigmoid(x2)
        x2 = self.inv_MixLogCDF(x2, ml_logits, ml_means, ml_logscales, self.n_components)

        if self.split == 'channel':
            return tf.concat([x1, x2], axis=-1)
        else:
            x = tf.stack([x1, x2], axis=3)
            return tf.reshape(x, (-1, self.H, self.W, self.C))

    def _forward_log_det_jacobian(self, x, context=None):
        if self.split == 'channel':
            x1, x2 = tf.split(x, 2, axis=-1)
        else:
            x = tf.reshape(x, (-1, self.H, self.W // 2, 2, self.C))
            x1, x2 = tf.unstack(x, axis=3)

        if self.split_state:
            x2, x1 = x1, x2

        log_s, t, ml_logits, ml_means, ml_logscales = self.nn(x1, context=context)

        log_det = self.MixLog_logPDF(x1, ml_logits, ml_means, ml_logscales, self.n_components)
        y2 = tf.exp(self.MixLog_logCDF(x1, ml_logits, ml_means, ml_logscales, self.n_components))
        y2 = tf.clip_by_value(y2, clip_value_min=float(1e-10), clip_value_max=float(1. - 1e-7))
        #assert tf.reduce_all(y2 < 1.), tf.reduce_max(y2)
        #assert tf.reduce_all(y2 > 0.), tf.reduce_min(y2)
        log_det += -tf.math.log(1. - y2) - tf.math.log(y2)
        log_det += log_s

        return tf.reduce_sum(log_det, axis=[1, 2, 3])

    def MixLog_logCDF(self, x, p, mu, log_s, n_components, min_log_s=-7.):
        log_s = tf.maximum(log_s, min_log_s)
        log_p = tf.nn.log_softmax(p, axis=-1)

        x = tf.expand_dims(x, axis=-1)
        # x = tf.repeat(x, n_components, axis=-1)

        # assert x.shape == p.shape == mu.shape == log_s.shape

        log_sig_x = tf.math.log_sigmoid((x - mu) * tf.exp(-log_s))
        z = log_p + log_sig_x

        return tf.reduce_logsumexp(z, axis=-1)

    def inv_MixLogCDF(self, y, p, mu, log_s, n_components,
                      position_tolerance=1e-8, value_tolerance=1e-8):

        with tf.control_dependencies([self.assert_in_range(y, min=0., max=1.)]):
            y = tf.identity(y)

        init_x = tf.zeros_like(y)

        def objective_fn(x):
            return y - tf.exp(self.MixLog_logCDF(x, p, mu, log_s, n_components))

        results = tfp.math.secant_root(objective_fn, initial_position=init_x,
                                       position_tolerance=position_tolerance,
                                       value_tolerance=position_tolerance)

        return results.estimated_root

    def MixLog_logPDF(self, x, p, mu, log_s, n_components, min_log_s=-7.):
        log_s = tf.maximum(log_s, min_log_s)
        log_p = tf.nn.log_softmax(p, axis=-1)

        x = tf.expand_dims(x, axis=-1)
        # x = tf.repeat(x, n_components, axis=-1)

        # assert x.shape == p.shape == mu.shape == log_s.shape

        scale_x = (x - mu) * tf.exp(-log_s)
        z = log_p + scale_x - log_s - 2 * tf.nn.softplus(scale_x)

        return tf.reduce_logsumexp(z, axis=-1)

    @staticmethod
    def inv_sigmoid(x):
        return -tf.math.log(tf.math.reciprocal(x) - 1.)

    @staticmethod
    def assert_in_range(x, *, min, max):
        """Asserts that x is in [min, max] elementwise"""
        return tf.Assert(tf.logical_and(
            tf.greater_equal(tf.reduce_min(x), min),
            tf.less_equal(tf.reduce_max(x), max)
        ), [x])
