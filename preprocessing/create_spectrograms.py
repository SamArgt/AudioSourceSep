import warnings
from scipy.io import wavfile
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from librosa.feature import melspectrogram
import os
import argparse
import numpy as np
import re
import time
warnings.filterwarnings("ignore")


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


def mel_spectrograms_from_gen(song_gen, sr, n_fft=2048, hop_length=512,
                              n_mels=128):
    """
    Take as input a generator of raw audio: array of shape (length, 2)
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


def main():
    """
    Walk the input folder
    For each wav file:
            Save spectrograms as .npy files into output directory

    """
    parser = argparse.ArgumentParser(
        description='Compute Mel spectrograms and save them')
    parser.add_argument('INPUT', type=str,
                        help='input dirpath of the wav files')
    parser.add_argument('OUTPUT', type=str,
                        help='output dirpath for saving the spectrograms')
    parser.add_argument('--params', type=str,
                        help='parameters for the computation: length_sec,stride,n_fft,hop_length,n_mels',
                        default=None)
    args = parser.parse_args()

    if args.params is None:
        length_sec = input(
            'Window length in seconds for each spectrograms (default 5): ')
        if length_sec == '':
            length_sec = 5
        else:
            length_sec = float(length_sec)
        stride = input('Stride of the window (default rate * length_sec): ')
        if stride == '':
            stride = None
        else:
            stride = int(stride)
        n_fft = input('Window size for the STFT (default 2048): ')
        if n_fft == '':
            n_fft = 2048
        else:
            n_fft = int(n_fft)
        hop_length = input('hop length for the STFT (default 512): ')
        if hop_length == '':
            hop_length = 512
        else:
            hop_length = int(hop_length)
        n_mels = input('number of mel frequencies (default 128): ')
        if n_mels == '':
            n_mels = 128
        else:
            n_mels = int(n_mels)
    else:
        length_sec, stride, n_fft, hop_length, n_mels = args.params.split(',')
        length_sec = float(length_sec)
        if stride == 'None':
            stride = None
        else:
            stride = int(stride)
        n_fft, hop_length, n_mels = int(n_fft), int(hop_length), int(n_mels)

    print("Mel Spectrograms parameters: \n length_sec={} \n stride={} \n n_fft={} \n hop length={} \n n mels={}"
          .format(length_sec, stride, n_fft, hop_length, n_mels))
    print('\n')

    t0 = time.time()

    input_dirpath = os.path.abspath(args.INPUT)
    output_dirpath = os.path.abspath(args.OUTPUT)

    wav_files = []
    for root, dirs, files in os.walk(input_dirpath):
        current_path = os.path.join(input_dirpath, root)
        if len(files) > 0:
            wav_files += [os.path.join(current_path, f)
                          for f in files if re.match(".*(.)wav$", f)]

    for wav_file in wav_files:
        song_gen, rate = load_wav(wav_file, length_sec, stride)
        melspectrograms = mel_spectrograms_from_gen(
            song_gen, rate, n_fft, hop_length, n_mels)
        # dirpath for the current wav file
        wav_file_relpath = os.path.relpath(wav_file, input_dirpath)
        temp_output_dirpath = os.path.join(output_dirpath, wav_file_relpath)
        temp_output_dirpath = temp_output_dirpath[:-4]
        wav_file_output = os.path.join(
            temp_output_dirpath, os.path.split(temp_output_dirpath)[1])
        try:
            os.makedirs(temp_output_dirpath)
        except FileExistsError:
            pass
        N = save_mel_spectrograms(melspectrograms, wav_file_output)

        print("File {} saved into {} spectrograms".format(wav_file_relpath, N))
    print("-" * 40)
    deltaT = np.round(time.time() - t0, 2)
    print("{} wav files transformed into spectrograms in {} seconds.".format(
        len(wav_files), deltaT))


if __name__ == '__main__':
    main()
