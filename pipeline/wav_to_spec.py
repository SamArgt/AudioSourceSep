from preprocessing import *
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


def main(args):
    """
    Walk the input folder
    For each wav file:
            Save spectrograms as .npy files into output directory

    """
    t0 = time.time()

    input_dirpath = os.path.abspath(args.INPUT)
    output_dirpath = os.path.abspath(args.OUTPUT)

    try:
        os.mkdir(output_dirpath)
    except FileExistsError:
        pass

    logfile = open(os.path.join(output_dirpath, "out.log"), 'w')
    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)
    logfile.write(template)

    wav_files = []
    for root, dirs, files in os.walk(input_dirpath):
        current_path = os.path.join(input_dirpath, root)
        if len(files) > 0:
            wav_files += [os.path.join(current_path, f)
                          for f in files if re.match(".*(.)wav$", f)]

    for wav_file in wav_files:
        # load the wav file
        song_ds, rate = load_wav(wav_file, args.length_sec, sr=args.sr)
        print('{} Loaded...'.format(wav_file))
        # compute the spectrograms
        if args.use_signal:
            melspectrograms_ds = mel_spectrograms_from_ds_tfSignal(song_ds, rate,
                                                                   int(rate * args.length_sec),
                                                                   args.n_fft, args.hop_length, args.n_mels,
                                                                   fmin=args.fmin, fmax=args.fmax, dbmin=args.dbmin,
                                                                   dbmax=args.dbmax)
            print("\t Mel Spectrograms computed using tf.signal")
        else:
            melspectrograms_ds = mel_spectrograms_from_ds(
                song_ds, rate, args.n_fft, args.hop_length, args.n_mels)
            print("\t Mel Spectrograms computed using librosa")

        # save the spectrograms
        filename_dirpath = os.path.join(output_dirpath, os.path.split(wav_file)[1])
        filename_dirpath = filename_dirpath[:-4]
        print
        if args.tfrecords:
            save_tf_records(melspectrograms_ds, filename_dirpath)
            print('\t Saved as tfrecords at {}'.format(filename_dirpath))
        else:
            N = save_mel_spectrograms(melspectrograms_ds, filename_dirpath)
            print("\tSaved into {} spectrograms as npy".format(N))

    print("-" * 40)
    deltaT = np.round(time.time() - t0, 2)
    print("{} wav files saved as spectrograms in {} seconds.".format(
        len(wav_files), deltaT))
    logfile.write("{} wav files saved as spectrograms in {} seconds.".format(
        len(wav_files), deltaT))
    logfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute Mel spectrograms and save them')
    parser.add_argument('INPUT', type=str,
                        help='input dirpath of the wav files')
    parser.add_argument('OUTPUT', type=str,
                        help='output dirpath for saving the spectrograms')

    parser.add_argument('--length_sec', type=float, default=2.04,
                        help='Window length in seconds for each spectrograms')
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling Rate. If None, keeps the sampling rate of the wav files")
    parser.add_argument("--n_fft", type=int, default=2048,
                        help="Window size of the STFT")
    parser.add_argument("--hop_length", type=int, default=512,
                        help="Hop Length of the STFT")
    parser.add_argument("--n_mels", type=int, default=96,
                        help="Number of mel frequencies")
    parser.add_argument('--fmin', type=int, default=125, help="Minimum frequency of the mel filter")
    parser.add_argument('--fmax', type=int, default=7600, help="Maximum frequency of the mel filter")
    parser.add_argument("--dbmin", type=int, default=-100, help="Minimum DB of the spectrogram")
    parser.add_argument("--dbmax", type=int, default=20, help="Maximum DB of the spectrogram")

    parser.add_argument('--use_signal', action="store_true",
                        help='Either to use tf.signal or not (otherwise use librosa)')

    parser.add_argument('--tfrecords', action="store_true",
                        help="Either to save as tfrecords or not (otherwise as npy)")
    args = parser.parse_args()
    main(args)
