from scipy.io import wavfile
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
import numpy as np
import tensorflw as tf
import re


def load_wav(path, length_sec, stride=None):
    """
    Load wav file from path
    Cut the audio in windows
    Return an iterator

    path (str) : path of the wav file
    length_sec (float, int) : size of the window in seconds
    stride (int) : stride between each window
        if stride is None: build non overlapping window

    Outputs:
    tensorflow TimeseriesGenerator of size (1,rate*length_sec, 2) (stereo)
    rate
    """
    rate, song = wavfile.read(path)
    song = np.array(song)
    LENGTH = int(rate * length_sec)
    if stride is None:
        STRIDE = int(LENGTH)
    else:
        STRIDE = int(stride)
    song_gen = TimeseriesGenerator(
        song, targets=song, length=LENGTH, stride=STRIDE, batch_size=1)
    return song_gen, rate


def load_wav_tf(path, length_sec, stride=None):
    """
    Load multiple wav files and prepare them to a tensorflow dataset

    path: directory of the wav files (can contain sub-directory)
    length_sec: length in second to frame the tracks
    stride: stride between each frame

    Output:
    tf.data.Dataset
    """
    path = os.path.abspath(path)
    wav_files = []
    for root, dirs, files in os.walk(path):
        current_path = os.path.join(path, root)
        if len(files) > 0:
            wav_files += [os.path.join(current_path, f)
                          for f in files if re.match(".*(.)wav$", f)]

    def gen():
        for i in range(song_gen.__len__()):
            yield(song_gen[i][0][0])
    dataset = None
    for wav_file in wav_files:
        song_gen, rate = load_wav(wav_file, length_sec, stride)
        temp_dataset = tf.data.Dataset.from_generator(
            gen, output_types=np.float64)
        if dataset is None:
            dataset = temp_dataset
        else:
            dataset = dataset.concatenate(temp_dataset)

    print("{} wav files loaded".format(len(wav_files)))

    return dataset


def load_spec(directory):
    """
    Load spectrograms from one directory into a tensorflow datasets
    spectrograms need to be save in .npy format

    directory: directory path of the .npy files

    Outputs:
    dataset: a tensorflow dataset
    """
    dataset = tf.data.Dataset.list_files(os.path.join(directory, "*.npy"))

    def load_npy_fn(t: tf.Tensor):
        return tf.constant(np.load(t.numpy()))

    dataset = dataset.map(lambda x: tf.py_function(
        func=load_npy_fn, inp=[x], Tout=tf.float64))

    return dataset


def load_spec_tf(directory):
    """
    Walk through the directory (and sub-directories) and load every .npy files
    Prepare them into a tensorflow dataset
    """
    path = os.path.abspath(directory)
    npy_files = []
    dataset = None
    for root, dirs, files in os.walk(path):
        current_path = os.path.join(path, root)
        temp_dataset = load_spec(current_path)
        if np.any([re.match(".*(.)npy$", f) is not None for f in files]):
            if temp_dataset is not None:
                if dataset is None:
                    dataset = temp_dataset
                else:
                    dataset = dataset.concatenate(temp_dataset)

    return dataset
