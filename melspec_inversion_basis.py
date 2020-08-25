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
    def griffin_inversion(melspec):
        if args.scale == "dB":
            melspec = librosa.db_to_power(melspec)
        return librosa.feature.inverse.mel_to_audio(melspec, sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length)
    return griffin_inversion


def stft_inversion_fn(sr=16000, fmin=125, fmax=7600, n_fft=2048, hop_length=512, scale="dB"):
    def stft_inversion(inputs):
        melspec, phase = inputs
        if args.scale == "dB":
            melspec = librosa.db_to_power(melspec)
        mel_stft = librosa.feature.inverse.mel_to_stft(melspec, sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft)
        stft_complex = complex_array(mel_stft, phase)
        istft = librosa.istft(stft_complex, hop_length=hop_length)
        return istft
    return stft_inversion

def main(args):

    sr = 16000
    fmin = 125
    fmax = 7600
    n_fft = 2048
    hop_length = 512

    os.chdir(args.basis_results)
    basis_results = np.load('results.npz')

    if args.output is None:
        args.output = 'inverse' + '_' + args.method
    if args.inverse_concat:
        args.output += '_inverse_concat'
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
    mixed_phase = basis_results['mixed_phase']

    assert len(x1.shape) == len(x2.shape) == len(mixed_phase.shape) == 3, (x1.shape, x2.shape, mixed_phase.shape)
    if (args.scale != 'dB') and (args.scale != 'power'):
        raise ValueError('scale should be dB or power')

    args.shape = x1.shape
    params_dict = vars(args)
    template = 'Spectrograms \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    if args.inverse_concat:
        x1 = np.concatenate(list(x1), axis=-1)
        x2 = np.concatenate(list(x2), axis=-1)
        mix = np.concatenate(list(mix), axis=-1)
        gt1 = np.concatenate(list(gt1), axis=-1)
        gt2 = np.concatenate(list(gt2), axis=-1)
        mixed_phase = np.concatenate(list(mixed_phase), axis=-1)

    if args.method == 'griffin':
        inversion_fn = griffin_inversion_fn(sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length, scale=args.scale)
    elif args.method == 'reuse_phase':
        inversion_fn = stft_inversion_fn(sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length, scale=args.scale)
        if args.inverse_concat:
            x1 = [x1, mixed_phase]
            x2 = [x2, mixed_phase]
            gt1 = [gt1, mixed_phase]
            gt2 = [gt2, mixed_phase]
            mix = [mix, mixed_phase]
        else:
            x1 = [[x1[i], mixed_phase[i]] for i in range(len(x1))]
            x2 = [[x2[i], mixed_phase[i]] for i in range(len(x2))]
            gt1 = [[gt1[i], mixed_phase[i]] for i in range(len(gt1))]
            gt2 = [[gt2[i], mixed_phase[i]] for i in range(len(gt2))]
            mix = [[mix[i], mixed_phase[i]] for i in range(len(mix))]
    else:
        raise ValueError('method should be griffin or reuse_phase')

    t_init = time.time()
    if args.inverse_concat:
        x1_inv = inversion_fn(x1)
        x2_inv = inversion_fn(x2)
        gt1_inv = inversion_fn(gt1)
        gt2_inv = inversion_fn(gt2)
        mix_inv = inversion_fn(mix)
    else:
        inv_spec = {'x1': [], 'x2': [], 'gt1': [], 'gt2': [], 'mix': []}
        for i in range(len(x1)):
            spec_to_invert = {'x1': x1[i], 'x2': x2[i], 'gt1': gt1[i], 'gt2': gt2[i], 'mix': mix[i]}
            print("Start inversing Spectrograms {} / {} at {}".format(i + 1, len(x1), datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")))
            for n, spec in spec_to_invert.items():
                t0 = time.time()
                spec_inv_i = inversion_fn(spec)
                inv_spec[n].append(spec_inv_i)
                print("Done melspec {} in {} seconds".format(n, round(time.time() - t0, 3)))

        x1_inv = np.concatenate(inv_spec['x1'], axis=-1)
        x2_inv = np.concatenate(inv_spec['x2'], axis=-1)
        gt1_inv = np.concatenate(inv_spec['gt1'], axis=-1)
        gt2_inv = np.concatenate(inv_spec['gt2'], axis=-1)
        mix_inv = np.concatenate(inv_spec['mix'], axis=-1)

    t1 = time.time()
    duration = round(t1 - t_init, 4)

    print("Inversion duration: {} seconds".format(duration))

    if args.save_wav:
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

    parser.add_argument("--method", type=str, default="griffin", help="griffin or reuse_phase")
    parser.add_argument('--inverse_concat', action="store_true", help="Inverse the concatenation of the Spectrograms")
    parser.add_argument("--scale", type=str, default="dB")
    parser.add_argument('--save_wav', action='store_true')

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    main(args)
