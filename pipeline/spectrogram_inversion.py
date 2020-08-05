import numpy as np
import librosa
import soundfile as sf
import argparse
import time
import os


def complex_array(amplitudes, angles):
    return amplitudes * np.exp(1j * angles)


def griffin_inversion_fn(sr=16000, fmin=125, fmax=7600):
    def griffin_inversion(melspec):
        return librosa.feature.inverse.mel_to_audio(melspec, sr=sr, fmin=fmin, fmax=fmax)
    return griffin_inversion


def stft_inversion_fn(phase, sr=16000, fmin=125, fmax=7600, n_fft=2048):
    def stft_inversion(melspec):
        mel_stft = librosa.feature.inverse.mel_to_stft(melspec, sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft)
        stft_complex = complex_array(mel_stft, phase)
        istft = librosa.istft(stft_complex)
        return istft
    return stft_inversion

def main(args):

    args.sr = 16000
    args.fmin = 125
    args.fmax = 7600
    args.n_fft = 2048

    basis_results = np.load(args.basis_results)

    x1 = basis_results['x1']
    x2 = basis_results['x2']
    mixed_phase = basis_results['mixed_phase']

    assert len(x1.shape) == len(x2.shape) == len(mixed_phase.shape) == 3, (x1.shape, x2.shape, mixed_phase.shape)
    args.shape = x1.shape
    params_dict = vars(args)
    template = 'Spectrograms \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    if args.inverse_concat:
        x1 = np.concatenate(list(x1), axis=-1)
        x2 = np.conatenate(list(x2), axis=-1)
        mixed_phase = np.concatenate(list(mixed_phase), axis=-1)

    if args.method == 'griffin':
        inversion_fn = griffin_inversion_fn(sr=args.sr, fmin=args.fmin, fmax=args.fmax)
    elif args.method == 'reuse_phase':
        inversion_fn = stft_inversion_fn(mixed_phase, sr=args.sr, fmin=args.fmin, fmax=args.fmax, n_fft=args.n_fft)
    else:
        raise ValueError('method should be griffin or reuse_phase')

    t0 = time.time()
    if args.inverse_concat:
        x1_inv = inversion_fn(x1)
        x2_inv = inversion_fn(x2)
    else:
        x1_inv = []
        x2_inv = []
        for i in range(len(x1)):
            x1_inv.append(inversion_fn(x1[i]))
            x2_inv.append(inversion_fn(x2[i]))
        x1_inv = np.array(x1_inv)
        x2_inv = np.array(x2_inv)
    t1 = time.time()
    duration = round(t1 - t0, 4)

    print("Inversion duration: {} seconds".format(duration))

    try:
        os.mkdir(args.output)
        os.chdir(args.output)
    except FileExistsError:
        os.chdir(args.output)

    np.savez("inverse_spectrograms", x1_audio=x1_inv, x2_audio=x2_inv)

    with open('out.log', 'w') as log_file:
        log_file.write(template)
        log_file.write('\n')
        log_file.write("Inversion duration: {} seconds".format(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Spectrograms Inversion')
    parser.add_argument('basis_results', type=str, default=None,
                        help='directory of basis_results')

    parser.add_argument('--output', type=str, default='basis_sep',
                        help='output dirpath for savings')

    parser.add_argument("--method", type=str, default="griffin")
    parser.add_argument('--')

    args = parser.parse_args()

    main(args)
