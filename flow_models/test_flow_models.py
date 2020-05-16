from flow_models import *
import unittest
import tensorflow as tf
import numpy as np


class TestShiftAndLogScaleResNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = [64, 64, 3]
        cls.batch_size = 32
        cls.inputs = tf.random.normal([cls.batch_size] + cls.input_shape)
        cls.NN = ShiftAndLogScaleResNet(
            input_shape=cls.input_shape, n_filters=16)

    def test_output_shape(self):
        log_s, t = self.NN(self.inputs)
        self.assertEqual(log_s.shape, self.inputs.shape)
        self.assertEqual(t.shape, self.inputs.shape)


def make_test_case_bijector(bijector_class, inputs, expected_log_det, **kwargs):
    class TestBijector(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.inputs = inputs
            cls.bijector = bijector_class(**kwargs)
            cls.expected_log_det = expected_log_det

        def test_inversibility(self):
            outputs = self.bijector.forward(self.inputs)
            inv_outputs = self.bijector.inverse(outputs)
            is_equal = np.all(self.inputs == inv_outputs)
            self.assertTrue(is_equal)

        def test_log_det(self):
            bij_forward_log_det = self.bijector.forward_log_det_jacobian(
                self.inputs, event_ndims=3).numpy()[0]
            outputs = self.bijector.forward(self.inputs)
            bij_inv_log_det = self.bijector.inverse_log_det_jacobian(
                outputs, event_ndims=3).numpy()[0]
            self.assertEqual(bij_forward_log_det, -bij_inv_log_det)
            self.assertEqual(bij_forward_log_det, self.expected_log_det)

    return TestBijector


global EVENT_SHAPE, EVENT_SHAPE_1, EVENT_SHAPE_2, EXPECTED_LOG_DET, INPUTS, INPUTS_1, INPUTS_2
EVENT_SHAPE = [2, 2, 1]
EVENT_SHAPE_1 = [4, 4, 1]
EVENT_SHAPE_2 = [2, 2, 2]
EXPECTED_LOG_DET = np.log(2, dtype=np.float32)
INPUTS = tf.random.normal([1] + EVENT_SHAPE)
INPUTS_1 = tf.random.normal([1] + EVENT_SHAPE_1)
INPUTS_2 = tf.random.normal([1] + EVENT_SHAPE_2)


def shift_and_log_scale_toy(x):
    log_s = EXPECTED_LOG_DET * tf.ones(x.shape, dtype=tf.float32)
    t = tf.ones(x.shape, dtype=tf.float32)
    return log_s, t


def shift_and_log_scale_layer_toy(event_shape, n_filters):
    return shift_and_log_scale_toy


affine_coupling_layers_checkboard_args = {"event_shape": EVENT_SHAPE,
                                          "shift_and_log_scale_fn": shift_and_log_scale_toy,
                                          "masking": "checkboard"}

affine_coupling_layers_channel_args = {"event_shape": EVENT_SHAPE_2,
                                       "shift_and_log_scale_fn": shift_and_log_scale_toy,
                                       "masking": "channel"}

real_nvp_step_checkboard_args = {'event_shape': EVENT_SHAPE,
                                 'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                                 'n_filters': 2, 'masking': 'checkboard'}

real_nvp_block_args = {'event_shape': EVENT_SHAPE,
                       'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                       'n_filters': 2}

real_nvp_bijector_args = {'input_shape': EVENT_SHAPE_1,
                          'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                          'n_filters_base': 2}


# One affine coupling layer with checkboard: Input dimension (2, 2, 1) ->
# 2 variables are scaled by log(2) -> the expected log det is  (2 * log(2))
class TestAffineCouplingLayersMaskedCheckboard(make_test_case_bijector(AffineCouplingLayerMasked,
                                                                       INPUTS,
                                                                       expected_log_det=2 * EXPECTED_LOG_DET,
                                                                       **affine_coupling_layers_checkboard_args)):
    pass


# One affine coupling layer with channel: Input dimension (2, 2, 2) ->
# 1 * 4 variables are scaled by log(2) -> the expected log det is  (4 * log(2))
class TestAffineCouplingLayersMaskedChannel(make_test_case_bijector(AffineCouplingLayerMasked,
                                                                    INPUTS_2,
                                                                    expected_log_det=4 * EXPECTED_LOG_DET,
                                                                    **affine_coupling_layers_channel_args)):
    pass


# 3 affine layers stacked: input dim = (2, 2, 1) -> expected log det = 3 * (2 * log2)
class TestRealNVPStepCheckboard(make_test_case_bijector(RealNVPStep, INPUTS,
                                                        expected_log_det=6 * EXPECTED_LOG_DET,
                                                        **real_nvp_step_checkboard_args)):
    pass


# 3 Real NVP Steps stacked: input dim = (2, 2, 1)) -> expected log det = 6 * (2 * log(2))
class TestRealNVPBlock(make_test_case_bijector(RealNVPBlock, INPUTS,
                                               expected_log_det=12 * EXPECTED_LOG_DET,
                                               **real_nvp_block_args)):
    pass


# 2 Real NVP Blocks stacked: input_dim = (4, 4, 1) -> expected log det = 12 * (8 * log(2))
class TestRealNVPBijector(make_test_case_bijector(RealNVPBijector, INPUTS_1,
                                                  expected_log_det=12 * 8 * EXPECTED_LOG_DET,
                                                  **real_nvp_bijector_args)):
    pass


if __name__ == '__main__':
    unittest.main()
