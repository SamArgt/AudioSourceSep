from flow_models.flow_tfp_bijectors import *
from flow_models.flow_glow import *
from flow_models.flow_real_nvp import *
from flow_models.flow_tfk_layers import *
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
            if self.expected_log_det is not None:
                bij_forward_log_det = self.bijector.forward_log_det_jacobian(
                    self.inputs, event_ndims=3).numpy()[0]
                outputs = self.bijector.forward(self.inputs)
                bij_inv_log_det = self.bijector.inverse_log_det_jacobian(
                    outputs, event_ndims=3).numpy()[0]
                self.assertEqual(bij_forward_log_det, -bij_inv_log_det)
                self.assertEqual(bij_forward_log_det, self.expected_log_det)
            else:
                pass

    return TestBijector


global EVENT_SHAPE, EVENT_SHAPE_1, EVENT_SHAPE_2, EVENT_SHAPE_3
global EXPECTED_LOG_DET, INPUTS, INPUTS_1, INPUTS_2, INPUTS_3
global MINIBATCH, MINIBATCH_1, MINIBATCH_2, MINIBATCh_3
EVENT_SHAPE = [2, 2, 1]
EVENT_SHAPE_1 = [4, 4, 1]
EVENT_SHAPE_2 = [2, 2, 2]
EVENT_SHAPE_3 = [8, 8, 1]
EXPECTED_LOG_DET = np.log(2, dtype=np.float32)
INPUTS = tf.random.normal([1] + EVENT_SHAPE)
INPUTS_1 = tf.random.normal([1] + EVENT_SHAPE_1)
INPUTS_2 = tf.random.normal([1] + EVENT_SHAPE_2)
INPUTS_3 = tf.random.normal([1] + EVENT_SHAPE_3)
MINIBATCH = tf.concat((2 * tf.ones((1, 2, 2, 1), dtype=tf.float32),
                       tf.ones((1, 2, 2, 1), dtype=tf.float32)), axis=0)
MINIBATCH_1 = tf.concat((2 * tf.ones((1, 4, 4, 1), dtype=tf.float32),
                         tf.ones((1, 4, 4, 1), dtype=tf.float32)), axis=0)
MINIBATCH_2 = tf.concat((2 * tf.ones((1, 2, 2, 2), dtype=tf.float32),
                         tf.ones((1, 2, 2, 2), dtype=tf.float32)), axis=0)
MINIBATCH_3 = tf.concat((2 * tf.ones([1] + EVENT_SHAPE_3, dtype=tf.float32),
                         tf.ones([1] + EVENT_SHAPE_3, dtype=tf.float32)), axis=0)


def shift_and_log_scale_toy(x):
    log_s = EXPECTED_LOG_DET * tf.ones(x.shape, dtype=tf.float32)
    t = tf.ones(x.shape, dtype=tf.float32)
    return log_s, t


def shift_and_log_scale_layer_toy(event_shape, n_hidden_units, dtype=tf.float32, name='toy', l2_reg=None):
    return shift_and_log_scale_toy


affine_coupling_layers_checkboard_args = {"event_shape": EVENT_SHAPE,
                                          "shift_and_log_scale_layer": shift_and_log_scale_layer_toy,
                                          'n_hidden_units': 2, "masking": "checkboard"}

affine_coupling_layers_channel_args = {"event_shape": EVENT_SHAPE_2,
                                       "shift_and_log_scale_layer": shift_and_log_scale_layer_toy,
                                       'n_hidden_units': 2, "masking": "channel"}

affine_coupling_layer_split_args = {"event_shape": EVENT_SHAPE_2,
                                    "shift_and_log_scale_layer": shift_and_log_scale_layer_toy,
                                    'n_hidden_units': 2}

act_norm_args = {'event_shape': EVENT_SHAPE,
                 'minibatch': MINIBATCH}

inv1x1conv_args = {'event_shape': EVENT_SHAPE_2}

glowstep_args = {'event_shape': EVENT_SHAPE_2,
                 'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                 'n_hidden_units': 2, 'minibatch': MINIBATCH_2}

glowblock_args = {'K': 2, 'event_shape': EVENT_SHAPE_1,
                          'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                          'n_hidden_units': 2, 'minibatch': MINIBATCH_1}

glowbijector2_args = {'K': 2, 'event_shape': EVENT_SHAPE_1,
                      'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                      'n_hidden_units': 2, 'minibatch': MINIBATCH_1}

glowbijector3_args = {'K': 2, 'event_shape': EVENT_SHAPE_3,
                      'shift_and_log_scale_layer': shift_and_log_scale_layer_toy,
                      'n_hidden_units': 2, 'minibatch': MINIBATCH_3}

preprocessing_args = {'event_shape': EVENT_SHAPE_1}


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


# One affine coupling layer splitting along the channel dimension: Input dimension (2, 2, 2) ->
# 1 * 4 variables are scaled by log(2) -> the expected log det is  (4 * log(2))
class TestAffineCouplingLayerSplit(make_test_case_bijector(AffineCouplingLayerSplit,
                                                           INPUTS_2,
                                                           expected_log_det=4 * EXPECTED_LOG_DET,
                                                           **affine_coupling_layer_split_args)):
    pass


# ActNorm layer: minibatch built such that scale_init = 2
# input_dim (2, 2, 1) -> expected_log_det = 2 * 2 *(1 * log(2))
class TestActNorm(make_test_case_bijector(ActNorm, INPUTS,
                                          expected_log_det=4 * EXPECTED_LOG_DET,
                                          **act_norm_args)):
    pass


# Test invertibility of Invertible1x1Conv
class TestInvertible1x1Conv(make_test_case_bijector(Invertible1x1Conv, INPUTS_2,
                                                    expected_log_det=None,
                                                    **inv1x1conv_args)):
    pass


# Test invertibility of GlowStep, GlowBlock and GlowBijector_2blocks
class TestGlowStep(make_test_case_bijector(GlowStep, INPUTS_2,
                                           expected_log_det=None,
                                           **glowstep_args)):
    pass


class TestGlowBlock(make_test_case_bijector(GlowBlock, INPUTS_1,
                                            expected_log_det=None,
                                            **glowblock_args)):
    pass


class TestGlowBijector_2Blocks(make_test_case_bijector(GlowBijector_2blocks, INPUTS_1,
                                                       expected_log_det=None,
                                                       **glowbijector2_args)):
    pass


class TestGlowBijector_3Blocks(make_test_case_bijector(GlowBijector_3blocks, INPUTS_3,
                                                       expected_log_det=None,
                                                       **glowbijector3_args)):
    pass


class TestPreprocessing(make_test_case_bijector(ImgPreprocessing, INPUTS_1,
                                                expected_log_det=None,
                                                **preprocessing_args)):
    pass


if __name__ == "__main__":
    unittest.main()
