import numpy as np
import librosa
import soundfile as sf
import argparse
import time
import os
import datetime
import sys


"""
Script for inversing estimated melspectograms from BASIS results

"""


def complex_array(amplitudes, angles):
    return amplitudes * np.exp(1j * angles)


def griffin_inversion_fn(sr=16000, fmin=125, fmax=7600, n_fft=2048, hop_length=512, scale="dB"):
    def griffin_inversion(melspecs):
        """
        Inverse melspectrograms with griffin algorithm

        Parameters:
            melspecs: list of ndarray

        Returns:
            i_melspecs: list of ndarray
        """

        i_melspecs = []
        for melspec in melspecs:
            if args.scale == "dB":
                melspec = librosa.db_to_power(melspec)
            i_melspecs.append(librosa.feature.inverse.mel_to_audio(melspec, sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length))
        return i_melspecs
    return griffin_inversion


def stft_inversion_fn(sr=16000, fmin=125, fmax=7600, n_fft=2048, hop_length=512, scale="dB", wiener_filter=False):
    def stft_inversion(inputs):
        """
        Inverse melspectrograms by reusing the phase

        Parameters:
            inputs: tuple
                (melspecs, stft_mixture)
                melspecs: list of ndarray
                    MelSpectrograms to invert
                stft_mixture: ndarray
                    STFT of the mixture to separate

        Returns:
            i_melspecs: list of ndarray
        """
        melspecs, stft_mixture = inputs
        n_src = len(melspecs)
        use_wiener_filter = wiener_filter and (n_src > 1)
        melspecs, stft_mixture = np.array(melspecs), np.array(stft_mixture)

        mel_stfts = []
        i_melspecs = []
        if args.scale == "dB":
            melspecs = librosa.db_to_power(melspecs)

        for i in range(len(melspecs)):
            mel_stft = librosa.feature.inverse.mel_to_stft(melspecs[i],
                                                           sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft)
            if use_wiener_filter:
                mel_stft = mel_stft ** 2

            mel_stfts.append(mel_stft)

        mel_stfts = np.array(mel_stfts)

        if use_wiener_filter:
            stft_complexs = single_channel_wiener_filter(mel_stfts, stft_mixture)

        for i in range(len(melspecs)):

            if use_wiener_filter:
                stft_complex = stft_complexs[i]
            else:
                stft_complex = complex_array(mel_stft[i], np.angle(stft_mixture))

            istft = librosa.istft(stft_complex, hop_length=hop_length)
            i_melspecs.append(istft)

        return i_melspecs

    return stft_inversion


def single_channel_wiener_filter(psd_sources, stft_mixture):
    """
    Perform Single Channel Wiener Filtering

    Parameters:
        psd_sources: list of ndarray
            power spectrograms of the estimated sources
        stft_mixture: nd.array, complex
            stft of the mixture

    Return:
        ndarray
            stft of the estimated sources
    """
    psd_sources = np.array(psd_sources)
    assert len(psd_sources.shape) == 3, psd_sources.shape
    assert psd_sources.shape[0] > 1, psd_sources.shape[0]
    try:
        stft_complexs = (psd_sources / (np.sum(psd_sources, axis=0) + 1e-10)) * stft_mixture
        return stft_complexs
    except ValueError:
        print(psd_sources.shape)
        print(np.sum(psd_sources, axis=0).shape)
        print(stft_mixture.shape)


def main(args):

    sr = 16000
    fmin = 125
    fmax = 7600
    n_fft = 2048
    hop_length = 512

    os.chdir(args.basis_results)
    basis_results = np.load('results.npz')

    if args.output is None:
        args.output = 'inverse' + '_' + args.algorithm + '_' + args.method
        if args.wiener_filter:
            args.output += '_wiener_filter'
    try:
        os.mkdir(args.output)
        os.chdir(args.output)
    except FileExistsError:
        os.chdir(args.output)
    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    x1 = basis_results['x1']
    x2 = basis_results['x2']
    gt1 = basis_results['gt1']
    gt2 = basis_results['gt2']
    mix = basis_results['mixed']
    stft_mixture = basis_results['stft_mixture']

    assert len(x1.shape) == len(x2.shape) == len(stft_mixture.shape) == 3, (x1.shape, x2.shape, stft_mixture.shape)
    if (args.scale != 'dB') and (args.scale != 'power'):
        raise ValueError('scale should be dB or power')

    args.shape = x1.shape
    params_dict = vars(args)
    template = 'Spectrograms \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    if args.method == 'whole':
        x1 = np.concatenate(list(x1), axis=-1)
        x2 = np.concatenate(list(x2), axis=-1)
        mix = np.concatenate(list(mix), axis=-1)
        gt1 = np.concatenate(list(gt1), axis=-1)
        gt2 = np.concatenate(list(gt2), axis=-1)
        stft_mixture = np.concatenate(list(stft_mixture), axis=-1)

    if args.algorithm == 'griffin':
        inversion_fn = griffin_inversion_fn(sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length, scale=args.scale)

        if args.method == 'whole':
            sources = [x1, x2]
            ground_truth = [gt1, gt2]
            mix = [mix]
        else:
            sources = [[x1[i], x2[i]] for i in range(len(x1))]
            ground_truth = [[gt1[i], gt2[i]] for i in range(len(gt1))]
            mix = [[mix[i]] for i in range(len(mix))]

    elif args.algorithm == 'reuse_phase':
        inversion_fn = stft_inversion_fn(sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length, scale=args.scale,
                                         wiener_filter=args.wiener_filter)
        if args.method == 'whole':
            sources = [[x1, x2], stft_mixture]
            ground_truth = [[x1, x2], stft_mixture]
            mix = [[mix], stft_mixture]
        else:
            sources = [[[x1[i], x2[i]], stft_mixture[i]] for i in range(len(x1))]
            ground_truth = [[[gt1[i], gt2[i]], stft_mixture[i]] for i in range(len(gt1))]
            mix = [[[mix[i]], stft_mixture[i]] for i in range(len(mix))]
    else:
        raise ValueError('method should be griffin or reuse_phase')

    t_init = time.time()
    if args.method == 'whole':
        x1_inv, x2_inv = inversion_fn([x1, x2])
        gt1_inv, gt2_inv = inversion_fn([gt1, gt2])
        mix_inv = inversion_fn([mix])[0]
    else:
        inv_spec = {'sources': [], 'ground_truth': [], 'mix': []}
        for i in range(len(x1)):
            spec_to_invert = {'sources': sources[i], 'ground_truth': ground_truth[i], 'mix': mix[i]}
            print("Start inversing Spectrograms {} / {} at {}".format(i + 1, len(x1), datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")))
            for n, spec in spec_to_invert.items():
                t0 = time.time()
                spec_inv_i = inversion_fn(spec)
                inv_spec[n].append(spec_inv_i)
                print("Done melspec {} in {} seconds".format(n, round(time.time() - t0, 3)))

        x1_inv = np.concatenate(np.array(inv_spec['sources'])[:, 0], axis=-1)
        x2_inv = np.concatenate(np.array(inv_spec['sources'])[:, 1], axis=-1)
        gt1_inv = np.concatenate(np.array(inv_spec['ground_truth'])[:, 0], axis=-1)
        gt2_inv = np.concatenate(np.array(inv_spec['ground_truth'])[:, 1], axis=-1)
        mix_inv = np.concatenate(np.array(inv_spec['mix'])[:, 0], axis=-1)

    t1 = time.time()
    duration = round(t1 - t_init, 4)

    print("Inversion duration: {} seconds".format(duration))

    sf.write("sep1.wav", data=x1_inv, samplerate=sr)
    sf.write("sep2.wav", data=x2_inv, samplerate=sr)
    sf.write("gt1.wav", data=gt1_inv, samplerate=sr)
    sf.write("gt2.wav", data=gt2_inv, samplerate=sr)
    sf.write("mix.wav", data=mix_inv, samplerate=sr)

    np.savez("inverse_spectrograms", x1_audio=x1_inv, x2_audio=x2_inv, gt1_audio=gt1_inv, gt2_audio=gt2_inv, mix_audio=mix_inv)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Spectrograms Inversion')
    parser.add_argument('basis_results', type=str, default=None,
                        help='directory of basis_results')

    parser.add_argument('--output', type=str, default=None,
                        help='output dirpath for savings')

    parser.add_argument("--algorithm", type=str, default="reuse_phase", help="griffin or reuse_phase")
    parser.add_argument('--method', type=str, help="frame or whole", default="frame")
    parser.add_argument("--scale", type=str, default="dB")
    parser.add_argument('--wiener_filter', action="store_true", help="Use Single Channel Wiener Filter as post-processing")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    main(args)
