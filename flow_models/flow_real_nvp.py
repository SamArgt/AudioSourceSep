import tensorflow as tf
import tensorflow_probability as tfp
from .flow_tfp_bijectors import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class RealNVP(tfb.Bijector):
    """
    Real NVP: 2 scales architecture
        3 coupling layers checkerboard - squeezing - 3 coupling layers channel
        Factor out half the dimension
        4 coupling layers checkerboard

    Parameters:
        event_shape (list of int)
        n_filters (int): number of filters in the first stack of coupling layers
        n_blocks (int): number of residual blocks in the coupling layers
    """

    def __init__(self, event_shape, n_filters=32, n_blocks=4, alpha=0.05):
        super(RealNVP, self).__init__(forward_min_event_ndims=3)
        self.H, self.W, self.C = event_shape[0], event_shape[1], event_shape[2]
        shift_and_log_scale_layer = ShiftAndLogScaleResNet
        # Preprocessing bijector
        self.preprocessing = ImgPreprocessing(event_shape, alpha=0.05)
        # First Scale: sequence of coupling-squeezing-coupling
        self.scale1_stack1 = StackedMaskedCouplingLayers(event_shape, 3, shift_and_log_scale_layer, 'checkerboard',
                                                         n_blocks=n_blocks, n_filters=n_filters)
        self.scale1_squeeze = Squeeze(event_shape)
        event_shape_new = self.scale1_squeeze.event_shape_out
        self.scale1_stack2 = StackedMaskedCouplingLayers(event_shape_new, 3, shift_and_log_scale_layer, 'channel',
                                                         n_blocks=n_blocks, n_filters=2 * n_filters)
        # Wrap these bijectors into one:
        self.scale1 = tfb.Chain([self.scale1_stack2, self.scale1_squeeze, self.scale1_stack1, self.preprocessing])
        # Second Scale after factoring out half the dimension
        event_shape_new = [self.H // 2, self.W // 2, self.C * 2]
        self.scale2_stack = StackedMaskedCouplingLayers(event_shape_new, 4, shift_and_log_scale_layer, 'checkerboard',
                                                        n_blocks=n_blocks, n_filters=2 * n_filters)

    def _forward(self, x):
        z1_h1 = self.scale1.forward(x)
        z1, h1 = tf.split(z1_h1, 2, axis=-1)
        z2 = self.scale2_stack.forward(h1)
        return tf.concat([z1, z2], axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.scale2_stack.inverse(z2)
        z1_h1 = tf.concat([z1, h1], axis=-1)
        x = self.scale1_stack2.inverse(z1_h1)
        x = self.scale1_squeeze.inverse(x)
        x = self.scale1_stack1.inverse(x)
        x = self.preprocessing.inverse(x)
        return x

    def _forward_log_det_jacobian(self, x):
        log_det = self.scale1.forward_log_det_jacobian(x, event_ndims=3)
        z1_h1 = self.scale1.forward(x)
        z1, h1 = tf.split(z1_h1, 2, axis=-1)
        log_det += self.scale2_stack.forward_log_det_jacobian(h1, event_ndims=3)
        return log_det

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
