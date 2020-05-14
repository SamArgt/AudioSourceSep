from flow_models import *
import unittest


class TestShiftAndLogScaleConvNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = [64, 64, 3]
        cls.batch_size = 32
        cls.inputs = tf.random.normal([cls.batch_size] + cls.input_shape)
        cls.NN = ShiftAndLogScaleConvNetGlow(input_shape=cls.input_shape, n_filters=16)

    def test_output_shape(self):

        log_s, t = self.NN(self.inputs)
        self.assertEqual(log_s.shape, self.inputs.shape)
        self.assertEqual(t.shape, self.inputs.shape)


class TestAffineCouplingLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.event_shape = [64, 64, 2]
        cls.NN_input_shape = [64, 64, 1]
        cls.batch_size = 32
        cls.inputs = tf.random.normal([cls.batch_size] + cls.event_shape)
        cls.NN = ShiftAndLogScaleConvNetGlow(
            input_shape=cls.NN_input_shape, n_filters=16)
        cls.layer = AffineCouplingLayerSplit(cls.NN)

    def test_forward_inverse_shape(self):
        self.assertEqual(self.layer.forward(self.inputs).shape,
                         self.inputs.shape)
        self.assertEqual(self.layer.inverse(self.inputs).shape,
                         self.inputs.shape)

    def test_log_det_jacobian_shape(self):
        self.assertEqual(self.layer.forward_log_det_jacobian(self.inputs,
                                                             event_ndims=3).shape,
                         self.batch_size)
        self.assertEqual(self.layer.inverse_log_det_jacobian(self.inputs,
                                                             event_ndims=3).shape,
                         self.batch_size)


class TestRealNVP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        event_shape = [64, 64, 2]
        n_filters = 32
        cls.model = RealNVP(event_shape, n_filters, batch_norm=False)
        cls.batch_size = 32
        cls.inputs = tf.random.normal([cls.batch_size] + event_shape)

    def test_output_shape(self):
        Z = self.model(self.inputs)
        self.assertEqual(Z.shape, self.inputs.shape)

    def test_log_prob_shape(self):
        self.assertEqual(self.model.flow.log_prob(self.inputs).shape,
                         self.batch_size)


if __name__ == '__main__':
    unittest.main()
