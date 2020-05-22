from pipeline.preprocessing import *
import unittest
import tensorflow as tf
import shutil
import os
import numpy as np


class TestLoadAndSaveTFRecords(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tensors1d = tf.random.normal((30, 2500))
        tensors2d = tf.random.normal((30, 64, 64))
        tensors3d = tf.random.normal((30, 32, 32, 3))
        cls.ds_1d = tf.data.Dataset.from_tensor_slices(tensors1d)
        cls.ds_2d = tf.data.Dataset.from_tensor_slices(tensors2d)
        cls.ds_3d = tf.data.Dataset.from_tensor_slices(tensors3d)

    def test_save_and_load(self):

        try:
            os.mkdir("test_files")
        except FileExistsError:
            pass

        save_tf_records(self.ds_1d, os.path.join('test_files', 'ds_1d_test.tfrecord'))
        save_tf_records(self.ds_2d, os.path.join('test_files', 'ds_2d_test.tfrecord'))
        save_tf_records(self.ds_3d, os.path.join('test_files', 'ds_3d_test.tfrecord'))

        loaded_ds_1d = load_tf_records(os.path.join('test_files', 'ds_1d_test.tfrecord'))
        loaded_ds_2d = load_tf_records(os.path.join('test_files', 'ds_2d_test.tfrecord'))
        loaded_ds_3d = load_tf_records(os.path.join('test_files', 'ds_3d_test.tfrecord'))

        shape_1d = list(loaded_ds_1d.take(1).as_numpy_iterator())[0].shape
        len_1d = len(list(loaded_ds_1d.as_numpy_iterator()))
        shape_2d = list(loaded_ds_2d.take(1).as_numpy_iterator())[0].shape
        len_2d = len(list(loaded_ds_2d.as_numpy_iterator()))
        shape_3d = list(loaded_ds_3d.take(1).as_numpy_iterator())[0].shape
        len_3d = len(list(loaded_ds_3d.as_numpy_iterator()))

        self.assertTrue(np.all(shape_1d == (2500,)))
        self.assertEqual(len_1d, 30)
        self.assertTrue(np.all(shape_2d == (64, 64)))
        self.assertEqual(len_2d, 30)
        self.assertTrue(np.all(shape_3d == (32, 32, 3)))
        self.assertEqual(len_3d, 30)

        shutil.rmtree('test_files')


if __name__ == '__main__':
    unittest.main()
