import tensorflow as tf
from librosa.feature import melspectrogram
import librosa
import os
import numpy as np
import re


def load_wav(path, length_sec, sr=None):
    """
    Load wav file from path
    Cut the audio in windows
    Return a generator

    path (str) : path of the wav file
    length_sec (float, int) : size of the window in seconds

    Outputs:
    tensorflow dataset
    """
    song, rate = librosa.core.load(path, sr=sr, mono=True)
    song = np.array(song)
    LENGTH = int(rate * length_sec)
    song_ds = tf.data.Dataset.from_tensor_slices(song)
    song_ds = song_ds.batch(LENGTH, drop_remainder=True)
    return song_ds, rate


def load_multiple_wav(path, length_sec):
    """
    Load multiple wav files and prepare them to a tensorflow dataset

    path: directory of the wav files (can contain sub-directory)
    length_sec: length in second to frame the tracks

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

    dataset = None
    for wav_file in wav_files:
        song_ds, rate = load_wav(wav_file, length_sec)
        if dataset is None:
            dataset = song_ds
        else:
            dataset = dataset.concatenate(song_ds)

    print("{} wav files loaded".format(len(wav_files)))

    return dataset


def mel_spectrograms_from_ds(song_ds, sr, n_fft=2048, hop_length=512,
                             n_mels=128, fmin=125, fmax=7600, dbmin=-100, dbmax=20):
    """
    Take as input a dataset of raw audio:

    Compute the mel spectrogram for each element

    Inputs:
    song_ds: tensorflow dataset
    n_fft (int): window size of the STFT
    sr (int): sampling rate of the raw audio
    hop_length (int): jump between each window of the STFT
    n_mel (int): number of mel frequencies

    Outputs:
    tensorflow datasets
    """

    def get_mel_spectrograms_fn(sr, n_fft=n_fft, hop_length=hop_length,
                                n_mels=n_mels, fmin=fmin, fmax=fmax):

        def get_mel_spectrograms(x):
            mel_spect = melspectrogram(y=np.asfortranarray(x), sr=sr, S=None, n_fft=n_fft,
                                       hop_length=hop_length, win_length=None, window='hann', center=True,
                                       pad_mode='reflect', power=2.0, n_mels=n_mels, fmin=fmin, fmax=fmax)
            return mel_spect

        return get_mel_spectrograms

    map_fn = get_mel_spectrograms_fn(sr, n_fft=n_fft, hop_length=hop_length,
                                     n_mels=n_mels)

    spect_dataset = song_ds.map(
        lambda x: tf.py_function(map_fn, inp=[x], Tout=tf.float32))

    powermin = np.exp(dbmin * np.log(10) / 10)
    powermax = np.exp(dbmax * np.log(10) / 10)
    spect_dataset = spect_dataset.map(lambda x: tf.clip_by_value(x, powermin, powermax))

    return spect_dataset


def mel_spectrograms_from_ds_tfSignal(song_ds, sr, frame_length, n_fft=2048, hop_length=512, n_mels=128):
    """
    Same function as above but use the tf.signal package

    Need to specify the frame_length (length_sec * rate)
    """
    ds_stft = song_ds.map(lambda x: tf.signal.stft(x,
                                                   frame_length=frame_length,
                                                   frame_step=hop_length,
                                                   fft_length=n_fft,
                                                   window_fn=tf.signal.hann_window,
                                                   pad_end=True, name=None))
    ds_stft_power = ds_stft.map(lambda x: tf.cast(tf.abs(x)**2, tf.float32))
    num_spectrogram_bins = (n_fft // 2) + 1
    A = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mels,
                                              num_spectrogram_bins=num_spectrogram_bins,
                                              sample_rate=sr,
                                              lower_edge_hertz=0.,
                                              upper_edge_hertz=float(sr) / 2,
                                              dtype=tf.dtypes.float32, name=None)
    ds_mel = ds_stft_power.map(lambda x: tf.matmul(x, A))
    return ds_mel


def save_mel_spectrograms(mel_spectrograms_ds, filename):
    """
    Save all the spectrograms as npy file

    Inputs:
    mel_spectrograms: tensorflow dataset
    filename (str): pathname or name to which the data is saved.

    Save N spectrograms with names: "filename_i"
    with i =  0,1,...,N and j = 0 or 1
    """
    count = 0
    for i, spect in enumerate(mel_spectrograms_ds):
        np.save(filename + '_{}'.format(i), np.array(spect))
        count += 1
    return count


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
        func=load_npy_fn, inp=[x], Tout=tf.float32))

    return dataset


def load_spec_tf(directory):
    """
    Walk through the directory (and sub-directories) and load every .npy files
    Prepare them into a tensorflow dataset
    """
    path = os.path.abspath(directory)
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


def _float_feature(value):
    """Returns a float_list from a float / double list/array/tensor"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(array):
    """
    Creates a tf.Example message ready to be written to a file.

    Parameters:
      array: array
          audio (raw or spectrograms)
    Returns:
     a tf.Example object
    """
    shape = list(array.shape)
    feature = {
        'array': _float_feature(np.reshape(array, -1)),
        'shape': _int64_feature(shape)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(array):
    tf_string = tf.py_function(
        serialize_example,
        [array],
        tf.string)
    return tf.reshape(tf_string, ())


def save_tf_records(dataset, filename):
    """
    Save a tensorflow dataset as a tf records

    Parameters:
        dataset: tensorflow dataset containing tensors
        It should be a dataset of tensors

        filename: str
        filename to save the tf.records
    """
    if not re.match(".*(.)tfrecord$", filename):
        filename += '.tfrecord'
    serialized_dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_dataset)
    return 0


def load_tf_records(filenames, dtype=tf.float32):
    """
    Load tf.records (saved with the above function) into a tensorflow dataset

    Parameters:
        filenames: list of the tf.records filenames

    Returns:
        tensorflow dataset
    """
    feature_description = {
        'array': tf.io.FixedLenSequenceFeature(shape=[], dtype=dtype, allow_missing=True),
        'shape': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.map(
        lambda x: tf.reshape(x['array'], x['shape']))

    return parsed_dataset
