import tensorflow as tf
from scipy.io import wavfile
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from librosa.feature import melspectrogram
import os
import numpy as np
import re


def load_wav(path, length_sec, stride=None):
    """
    Load wav file from path
    Cut the audio in windows
    Return a generator

    path (str) : path of the wav file
    length_sec (float, int) : size of the window in seconds
    stride (int) : stride between each window
        if stride is None: build non overlapping window

    Outputs:
    tensorflow TimeseriesGenerator of size (1,rate*length_sec, 2) (stereo)
    rate of the wav file
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


def mel_spectrograms_from_gen(song_gen, sr, n_fft=2048, hop_length=512,
                              n_mels=128):
    """
    Take as input a generator of raw audio: tuple of array of shape ((length, 2), _)
    Compute the mel spectrogram for each array

    Inputs:
    song_gen: tensorflow TimeseriesGenerator
    n_fft (int): window size of the STFT
    sr (int): sampling rate of the raw audio
    hop_length (int): jump between each window of the STFT
    n_mel (int): number of mel frequencies

    Outputs:
    Array of shape (N, 2, n_mel, length/hop_length)
    where N is the number of array in the iterator
     and length the length of each array
    """
    N = len(song_gen)
    length = song_gen[0][0].shape[1]
    time_length = int(np.round(length / hop_length, 0))
    mel_spectrograms = np.zeros((N, 2, n_mels, time_length))
    for i, (song, _) in enumerate(song_gen):
        song = song.reshape((-1, 2))
        song_left = np.asfortranarray(song[:, 0])
        song_right = np.asfortranarray(song[:, 1])
        mel_spect_left = melspectrogram(y=np.asfortranarray(song_left), sr=sr, S=None, n_fft=n_fft,
                                        hop_length=hop_length, win_length=None, window='hann', center=True,
                                        pad_mode='reflect', power=2.0, n_mels=n_mels)
        mel_spect_right = melspectrogram(y=np.asfortranarray(song_right), sr=sr, S=None, n_fft=n_fft,
                                         hop_length=hop_length, win_length=None, window='hann', center=True,
                                         pad_mode='reflect', power=2.0, n_mels=n_mels)

        mel_spectrograms[i, :, :, :] = np.array(
            [mel_spect_left, mel_spect_right])

    return mel_spectrograms


def save_mel_spectrograms(mel_spectrograms, filename):
    """
    Save all the spectrograms as npy file

    Inputs:
    mel_spectrograms: array of shape (N, 2, n_mel, time_length)
    filename (str): pathname or name to which the data is saved.

    Save N spectrograms with names: "filename_i"
    with i =  0,1,...,N and j = 0 or 1
    """
    for i, spect in enumerate(mel_spectrograms):
        np.save(filename + '_{}'.format(i), spect)
    return len(mel_spectrograms)


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
