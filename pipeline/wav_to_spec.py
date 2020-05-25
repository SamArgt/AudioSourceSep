from preprocessing import *
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


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
    parser.add_argument('--use_signal', action="store_true",
                        help='Either to use tf.signal or not (otherwise use librosa)')
    parser.add_argument('--tfrecords', action="store_true",
                        help="Either to save as tfrecords or not (otherwise as npy)")
    args = parser.parse_args()

    if args.params is None:
        length_sec = input(
            'Window length in seconds for each spectrograms (default 5): ')
        if length_sec == '':
            length_sec = 5
        else:
            length_sec = float(length_sec)
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
        length_sec, n_fft, hop_length, n_mels = args.params.split(',')
        length_sec = float(length_sec)
        n_fft, hop_length, n_mels = int(n_fft), int(hop_length), int(n_mels)

    print("\nMel Spectrograms parameters: \n length_sec={} \n n_fft={} \n hop length={} \n n mels={}"
          .format(length_sec, n_fft, hop_length, n_mels))
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
        # load the wav file
        song_ds, rate = load_wav(wav_file, length_sec)
        print('{} Loaded...'.format(wav_file))
        # compute the spectrograms
        if args.use_signal:
            melspectrograms_ds = mel_spectrograms_from_ds_tfSignal(song_ds, rate,
                                                                   int(rate * length_sec),
                                                                   n_fft, hop_length, n_mels)
            print("\t Mel Spectrograms computed using tf.signal")
        else:
            melspectrograms_ds = mel_spectrograms_from_ds(
                song_ds, rate, n_fft, hop_length, n_mels)
            print("\t Mel Spectrograms computed using librosa")

        # save the spectrograms
        wav_file_relpath = os.path.relpath(wav_file, input_dirpath)
        temp_output_dirpath = os.path.join(output_dirpath, wav_file_relpath)
        temp_output_dirpath = temp_output_dirpath[:-4]
        if args.tfrecords:
            try:
                os.makedirs(os.path.split(temp_output_dirpath)[0])
            except FileExistsError:
                pass
            save_tf_records(melspectrograms_ds, temp_output_dirpath)
            print('\t Saved as tfrecords')
        else:
            wav_file_output = os.path.join(
                temp_output_dirpath, os.path.split(temp_output_dirpath)[1])
            try:
                os.makedirs(temp_output_dirpath)
            except FileExistsError:
                pass

            else:
                N = save_mel_spectrograms(melspectrograms_ds, wav_file_output)
            print("\tSaved into {} spectrograms as npy".format(N))

    print("-" * 40)
    deltaT = np.round(time.time() - t0, 2)
    print("{} wav files saved as spectrograms in {} seconds.".format(
        len(wav_files), deltaT))


if __name__ == '__main__':
    main()
