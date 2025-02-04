import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import flow_builder
from datasets import data_loader
from librosa.display import specshow
import librosa
import train_utils
import argparse
import time
import os
import sys
import matplotlib.pyplot as plt
import soundfile as sf
from train_utils import *
from ncsn.utils import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


"""
Script for running the BASIS algorithm with pre-trained Glow or NCSN model

"""


def restore_checkpoint(ckpt, restore_path, model, optimizer, latest=True):
    if latest:
        checkpoint_restore_path = tf.train.latest_checkpoint(restore_path)
        assert restore_path is not None, restore_path
    else:
        checkpoint_restore_path = restore_path
    # Restore weights if specified
    status = ckpt.restore(checkpoint_restore_path)
    status.assert_existing_objects_matched()

    return ckpt


def image_grid(n_display, x, y, z, data_type="image", separation=True, **kwargs):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(nrows=n_display, ncols=3, figsize=(6, 8))
    if data_type == 'image' and x.shape[-1] == 1:
        cmap = 'binary'
    else:
        cmap = None
    for i in range(n_display):
        ax1, ax2, ax3 = axes[i]
        if data_type == "image":
            ax1.imshow(x[i].squeeze(), cmap=cmap)
            ax2.imshow(y[i].squeeze(), cmap=cmap)
            ax3.imshow(z[i].squeeze(), cmap=cmap)
            ax1.set_axis_off()
            ax2.set_axis_off()
            ax3.set_axis_off()
        else:
            specshow(x[i].squeeze(), sr=kwargs["sampling_rate"],
                     ax=ax1, x_axis='off', y_axis='off', fmin=kwargs["fmin"], fmax=kwargs["fmax"])
            specshow(y[i].squeeze(), sr=kwargs["sampling_rate"],
                     ax=ax2, x_axis='off', y_axis='off', fmin=kwargs["fmin"], fmax=kwargs["fmax"])
            specshow(z[i].squeeze(), sr=kwargs["sampling_rate"],
                     ax=ax3, x_axis='off', y_axis='off', fmin=kwargs["fmin"], fmax=kwargs["fmax"])

    if separation:
        title = "Separation: Mixture = Component 1 + Component 2"
    else:
        title = "Mixing: Component 1 + Component 2 = Mixture"
    f.suptitle(title)
    return f


@tf.function
def compute_grad_logprob(inputs, model):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        loss = model.log_prob(inputs)
    gradients = tape.gradient(loss, inputs)
    return gradients


def post_processing_fn(args):
    def post_processing(x):
        if args.use_logit:
            x = 1. / (1. + np.exp(-x))
            x = (x - args.alpha) / (1. - 2. * args.alpha)
        x = x * (args.maxval - args.minval) + args.minval
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
            if args.scale == "power":
                x = librosa.power_to_db(x)
        return x
    return post_processing


def spectrogram_inversion(melspec, sr, fmin, fmax, use_db=True):
    if use_db:
        melspec = librosa.db_to_power(melspec)
    inv_melspec = librosa.feature.inverse.mel_to_audio(melspec, sr=sr, fmin=fmin, fmax=fmax)
    return inv_melspec


def mixing_process(args):
    if args.data_type == 'image':
        def g(*sources):
            sources = tf.stack(sources, axis=0)
            return tf.reduce_mean(sources, axis=0, dtype=np.float32)

        def grad_g(*sources):
            K = len(sources)
            sources = tf.stack(sources, axis=0)
            grad_sources = tf.ones_like(sources, dtype=np.float32) / float(K)
            return tf.unstack(grad_sources, K, axis=0)

    else:
        if args.scale == 'power':
            def g(*sources):
                sources = tf.stack(sources, axis=0)
                return tf.reduce_mean(tf.math.sqrt(sources), axis=0, dtype=np.float32)**2

            def grad_g(*sources):
                K = len(sources)
                sources = tf.stack(sources, axis=0)
                grad_sources = (1 / (tf.math.sqrt(sources) + 1e-8))
                grad_sources *= tf.reduce_mean(tf.math.sqrt(sources), axis=0, dtype=np.float32, keepdims=True) ** 2
                return tf.unstack(grad_sources, K, axis=0)
        else:
            def g(*sources):
                K = len(sources)
                sources = tf.stack(sources, axis=0)
                # if sum in amplitude:
                # mixing = (20. / tf.math.log(10.)) * (tf.math.reduce_logsumexp(sources * tf.math.log(10.) / 20., axis=0) - tf.math.log(float(K)))
                # if sum in power:
                mixing = (10. / tf.math.log(10.)) * (tf.math.reduce_logsumexp(sources * tf.math.log(10.) / 10., axis=0) - tf.math.log(float(K)))
                return mixing

            def grad_g(*sources):
                K = len(sources)
                sources = tf.stack(sources, axis=0)
                # if sum in amplitude:
                # grad_sources = tf.nn.softmax(sources * tf.math.log(10.) / 20., axis=0)
                # if sum in power:
                grad_sources = tf.nn.softmax(sources * tf.math.log(10.) / 10., axis=0)
                return tf.unstack(grad_sources, K, axis=0)

    return g, grad_g


def basis_inner_loop(mixed, x1, x2, model1, model2, sigma_idx, sigmas, g, grad_g, post_processing,
                     model_type='ncsn', delta=2e-5, T=100, debug=True,
                     train_summary_writer=None, step=None, **kwargs):

    full_data_shape = list(mixed.shape)
    n_mixed = full_data_shape[0]
    sigma = sigmas[sigma_idx]
    sigmaL = sigmas[-1]
    eta = tf.constant(delta * (sigma / sigmaL) ** 2, dtype=tf.float32)
    lambda_recon = 1.0 / (sigma ** 2)
    for t in range(T):
        epsilon1 = tf.math.sqrt(2. * eta) * tf.random.normal(full_data_shape, dtype=tf.float32)
        epsilon2 = tf.math.sqrt(2. * eta) * tf.random.normal(full_data_shape, dtype=tf.float32)

        if model_type == 'ncsn':
            inputs1 = [x1, tf.ones(shape=(n_mixed,), dtype=tf.int32) * sigma_idx]
            inputs2 = [x2, tf.ones(shape=(n_mixed,), dtype=tf.int32) * sigma_idx]
            grad_logprob1 = model1(inputs1, training=True)
            grad_logprob2 = model2(inputs2, training=True)
        else:
            inputs1 = x1
            inputs2 = x2
            grad_logprob1 = compute_grad_logprob(inputs1, model1)
            grad_logprob2 = compute_grad_logprob(inputs2, model2)

        mixing = g(x1, x2)
        grad_mixing_x1, grad_mixing_x2 = grad_g(x1, x2)

        x1 = x1 + eta * (grad_logprob1 + lambda_recon * grad_mixing_x1 * (mixed - mixing)) + epsilon1
        x2 = x2 + eta * (grad_logprob2 + lambda_recon * grad_mixing_x2 * (mixed - mixing)) + epsilon2

        if debug:
            print('step : {} / {}'.format(t, T))
            assert bool(tf.math.is_nan(grad_logprob1).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(grad_logprob2).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(grad_mixing_x1).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(grad_mixing_x2).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(mixing).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(x1).numpy().any()) is False, (sigma, t)
            assert bool(tf.math.is_nan(x2).numpy().any()) is False, (sigma, t)

        if (train_summary_writer is not None) and (t % (T // 5) == 0):
            print('step : {} / {}'.format(t, T))
            if debug:
                x1_np = x1.numpy()
                x2_np = x2.numpy()
                grad_mixing_x1_np = grad_mixing_x1.numpy()
                grad_mixing_x2_np = grad_mixing_x2.numpy()
                print("x1 stats: mean = {} \t std = {} \t min = {} \t max = {}".format(x1_np.mean(), x1_np.std(), x1_np.min(), x1_np.max()))
                print("x2 stats: mean = {} \t std = {} \t min = {} \t max = {}".format(x2_np.mean(), x2_np.std(), x2_np.min(), x2_np.max()))
                print("grad_mixing_x1 stats: mean = {} \t std = {} \t min = {} \t max = {}".format(grad_mixing_x1_np.mean(), grad_mixing_x1_np.std(),
                                                                                                   grad_mixing_x1_np.min(), grad_mixing_x1_np.max()))
                print("grad_mixing_x1 stats: mean = {} \t std = {} \t min = {} \t max = {}".format(grad_mixing_x2_np.mean(), grad_mixing_x2_np.std(),
                                                                                                   grad_mixing_x2_np.min(), grad_mixing_x2_np.max()))
            with train_summary_writer.as_default():
                sample_mix = post_processing(mixed.numpy())
                sample_x1 = post_processing(x1.numpy())
                sample_x2 = post_processing(x2.numpy())
                figure = image_grid(5, sample_mix, sample_x1, sample_x2, separation=True, **kwargs)
                tf.summary.image("Components", train_utils.plot_to_image(figure),
                                 max_outputs=50, step=step + t)

    return x1, x2


def basis_outer_loop(mixed, x1, x2, model1, model2, optimizer, sigmas,
                     ckpt1, ckpt2, args, train_summary_writer):

    step = 0
    post_processing = post_processing_fn(args)
    g, grad_g = mixing_process(args)

    x_arr = {'x1': [x1.numpy()], 'x2': [x2.numpy()]}

    for sigma_idx, sigma in enumerate(sigmas):
        print("Sigma = {} ({} / {})".format(sigma, sigma_idx + 1, len(sigmas)))
        if args.model_type == 'glow':
            restore_path_1 = args.restore_dict_1[sigma]
            restore_checkpoint(ckpt1, restore_path_1, model1, optimizer)
            print("Model 1 at noise level {} restored from {}".format(sigma, restore_path_1))
            restore_path_2 = args.restore_dict_2[sigma]
            restore_checkpoint(ckpt2, restore_path_2, model2, optimizer)
            print("Model 2 at noise level {} restored from {}".format(sigma, restore_path_2))
        else:
            pass

        x1, x2 = basis_inner_loop(mixed, x1, x2, model1, model2, sigma_idx, sigmas, g, grad_g, post_processing,
                                  model_type=args.model_type, delta=2e-5, T=args.T, debug=args.debug,
                                  train_summary_writer=train_summary_writer, step=step * args.T,
                                  data_type=args.data_type, fmin=args.fmin, fmax=args.fmax, sampling_rate=args.sampling_rate)

        x_arr['x1'].append(x1.numpy())
        x_arr['x2'].append(x2.numpy())

        step += 1
        with train_summary_writer.as_default():

            sample_mix = post_processing(mixed.numpy())
            sample_x1 = post_processing(x1.numpy())
            sample_x2 = post_processing(x2.numpy())
            figure = image_grid(5, sample_mix, sample_x1, sample_x2, separation=True,
                                data_type=args.data_type, fmin=args.fmin, fmax=args.fmax, sampling_rate=args.sampling_rate)
            tf.summary.image("Components", train_utils.plot_to_image(figure),
                             max_outputs=50, step=step * args.T)

        print("inner loop done")
        print("_" * 100)

    return x1, x2, x_arr


def main(args):

    # noise conditionned models
    abs_restore_path_1 = os.path.abspath(args.RESTORE1)
    abs_restore_path_2 = os.path.abspath(args.RESTORE2)

    if args.config is not None:
        new_args = get_config(args.config)
        new_args.dataset = args.dataset
        new_args.debug = args.debug
        new_args.output = args.output
        new_args.song_dir = args.song_dir
        new_args.inverse = args.inverse
        new_args.model_type = args.model_type
        new_args.n_mixed = args.n_mixed
        args = new_args

    sigmas = get_sigmas(args.sigma1, args.sigmaL, args.num_classes, progression=args.progression)
    sigmas_tf = tf.constant(sigmas, dtype=tf.float32)

    if args.model_type == "glow":
        args.restore_dict_1 = {sigma: os.path.join(abs_restore_path_1, "sigma_" + str(round(sigma, 2)), "tf_ckpts") for sigma in sigmas}
        args.restore_dict_2 = {sigma: os.path.join(abs_restore_path_2, "sigma_" + str(round(sigma, 2)), "tf_ckpts") for sigma in sigmas}
    elif args.model_type == "ncsn":
        args.restore_dict_1 = args.restore_dict_2 = None
    else:
        raise ValueError("model_type should be 'ncsn' or 'glow'")

    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.data_type = "image"
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.data_type = "image"
    else:
        if args.song_dir is None:
            raise ValueError('song_dir is None')
        song_dir_abspath = os.path.abspath(args.song_dir)
        args.data_shape = [args.height, args.width, 1]
        args.data_type = "melspec"

    try:
        os.mkdir(args.output)
        os.chdir(args.output)
    except FileExistsError:
        os.chdir(args.output)

    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    # set up tensorboard
    train_summary_writer, test_summary_writer = train_utils.setUp_tensorboard()

    # get mixture
    t0 = time.time()
    if args.data_type == "image":
        mixed, x1, x2, gt1, gt2, minibatch = data_loader.get_mixture_toydata(dataset=args.dataset, n_mixed=args.n_mixed,
                                                                             use_logit=args.use_logit, alpha=args.alpha,
                                                                             noise=None, mirrored_strategy=None)
        args.minval = 0.
        args.maxval = 256.
        args.sampling_rate, args.fmin, args.fmax = None, None, None
    else:
        if args.song_dir is None:
            raise ValueError("song directory path is None")

        args.fmin = 125
        args.fmax = 7600
        args.sampling_rate = 16000
        if args.scale == 'power':
            args.maxval = 100.
            args.minval = 1e-10
        elif args.scale == 'dB':
            args.maxval = 20.
            args.minval = -100.
        else:
            raise ValueError("scale should be 'power' or 'dB'")

        mix_path = os.path.join(song_dir_abspath, 'mix.wav')
        piano_path = os.path.join(song_dir_abspath, 'piano.wav')
        violin_path = os.path.join(song_dir_abspath, 'violin.wav')
        args.use_dB = (args.scale == 'dB')
        spec_params = {'length_sec': 2.04, 'dbmin': -100, 'dbmax': 20, 'fmin': 125,
                       'fmax': 7600, 'use_dB': args.use_dB, 'n_fft': 2048,
                       'hop_length': 512, 'n_mels': 96, 'sr': 16000}
        duration = 2.04 * args.n_mixed

        mel_spec, raw_audio, stft_mixture = data_loader.get_song_extract(mix_path, piano_path, violin_path, duration, **spec_params)

        mixed, gt1, gt2 = mel_spec[0], mel_spec[1], mel_spec[2]
        # preprocessing mixture
        mixed = (mixed - args.minval) / (args.maxval - args.minval)
        if args.use_logit:
            mixed = mixed * (1. - 2 * args.alpha) + args.alpha
            mixed = tf.math.log(mixed) - tf.math.log(1. - mixed)

        x1 = tf.random.uniform(mixed.shape, dtype=tf.float32)
        x2 = tf.random.uniform(mixed.shape, dtype=tf.float32)

        # wiener filter:
        # p_x1, p_x2 = tf.unstack(tf.nn.softmax(tf.stack([x1, x2], axis=0), axis=0), axis=0)
        # x1 = p_x1 * mixed
        # x2 = p_x2 * mixed

    print("Data Loaded in {} seconds".format(round(time.time() - t0, 3)))

    # post_processing
    post_processing = post_processing_fn(args)

    # display originals
    with train_summary_writer.as_default():
        mix_post_process = post_processing(mixed.numpy())
        figure = image_grid(5, gt1.numpy(), gt2.numpy(), mix_post_process, data_type=args.data_type,
                            separation=False, fmin=args.fmin, fmax=args.fmax, sampling_rate=args.sampling_rate)
        tf.summary.image("Originals", train_utils.plot_to_image(figure), max_outputs=1, step=0)
        if args.data_type == "melspec":
            tf.summary.audio("Original Audio", np.reshape(raw_audio, (3, -1, 1)), sample_rate=args.sampling_rate, encoding='wav', step=0)
            sf.write("ground_truth1.wav", data=raw_audio[1], samplerate=args.sampling_rate)
            sf.write("ground_truth2.wav", data=raw_audio[2], samplerate=args.sampling_rate)
            sf.write("mix.wav", data=raw_audio[0], samplerate=args.sampling_rate)

    # build model
    if args.model_type == "glow":
        model1 = flow_builder.build_glow(minibatch, L=args.L, K=args.K, n_filters=args.n_filters, dataset=args.dataset,
                                         l2_reg=args.l2_reg, mirrored_strategy=None)
        model2 = flow_builder.build_glow(minibatch, L=args.L, K=args.K, n_filters=args.n_filters, dataset=args.dataset,
                                         l2_reg=args.l2_reg, mirrored_strategy=None)
    else:
        if args.version == 'v1':
            model1 = get_uncompiled_model(args, name="model1")
            model2 = get_uncompiled_model(args, name="model2")
        else:
            model1 = get_uncompiled_model_v2(args, sigmas=sigmas_tf, name="model1")
            model2 = get_uncompiled_model_v2(args, sigmas=sigmas_tf, name="model2")

    # set up optimizer
    optimizer = train_utils.setUp_optimizer(None, args)
    # checkpoint
    ckpt1, _ = train_utils.setUp_checkpoint(None, model1, optimizer)
    ckpt2, _ = train_utils.setUp_checkpoint(None, model2, optimizer)
    if args.model_type == "ncsn":
        restore_checkpoint(ckpt1, abs_restore_path_1, model1, optimizer, latest=False)
        print("Model 1 restore from {}".format(abs_restore_path_1))
        restore_checkpoint(ckpt2, abs_restore_path_2, model2, optimizer, latest=False)
        print("Model 2 restored from {}".format(abs_restore_path_2))

    # print parameters
    params_dict = vars(args)
    template = 'BASIS Separation \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)
    with train_summary_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)
    # run BASIS separation
    t0 = time.time()
    x1, x2, x_arr = basis_outer_loop(mixed, x1, x2, model1, model2, optimizer, sigmas,
                                     ckpt1, ckpt2, args, train_summary_writer)

    t1 = time.time()
    print("Duration: {} seconds".format(round(t1 - t0, 3)))

    # Save results
    x1 = post_processing(x1.numpy().squeeze())
    x2 = post_processing(x2.numpy().squeeze())
    mixed = post_processing(mixed.numpy().squeeze())
    gt1 = gt1.numpy().squeeze()
    gt2 = gt2.numpy().squeeze()
    x_arr['x1'] = post_processing(np.array(x_arr['x1']))
    x_arr['x2'] = post_processing(np.array(x_arr['x2']))
    np.savez('results', x1=x1, x2=x2, gt1=gt1.squeeze(), gt2=gt2.squeeze(), mixed=mixed, stft_mixture=stft_mixture)
    np.savez('results_convergence', x1=x_arr['x1'], x2=x_arr['x2'])

    # Inverse mel spec
    if args.data_type == "melspec" and args.inverse:
        x1_concat = np.concatenate(list(x1), axis=-1)
        x2_concat = np.concatenate(list(x2), axis=-1)
        x1_audio = spectrogram_inversion(x1_concat, sr=args.sampling_rate, fmin=args.fmin, fmax=args.fmax, use_db=args.use_dB)
        x2_audio = spectrogram_inversion(x2_concat, sr=args.sampling_rate, fmin=args.fmin, fmax=args.fmax, use_db=args.use_dB)
        sep_audio = np.reshape(np.array([x1_audio, x2_audio]), (2, -1, 1))
        sf.write("sep1.wav", data=x1_audio, samplerate=args.sampling_rate)
        sf.write("sep2.wav", data=x2_audio, samplerate=args.sampling_rate)
        with train_summary_writer.as_default():
            tf.summary.audio("Separated Audio", sep_audio, sample_rate=args.sampling_rate, encoding='wav', step=1000)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BASIS Separatation')
    parser.add_argument('RESTORE1', type=str, default=None,
                        help='directory of saved model1')
    parser.add_argument('RESTORE2', type=str, default=None,
                        help='directory of saved model2')

    parser.add_argument('--output', type=str, default='basis_sep',
                        help='output dirpath for savings')
    parser.add_argument('--debug', action="store_true")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="melspec",
                        help="mnist or cifar10 or melspec")

    # song directory path to separate
    parser.add_argument("--song_dir", type=str, default=None,
                        help="song directory path to separate: should contain\
                        3 songs: mix.wav, piano.wav and violin.wav")

    parser.add_argument("--inverse", action="store_true", help="Inverse spectrograms")
    # Model type
    parser.add_argument("--model_type", type=str, default="ncsn")

    parser.add_argument('--n_mixed', type=int, default=30,
                        help="number of mixture to separate")
    # config
    parser.add_argument('--config', type=str, help='path to the config file. Overwrite all other parameters below')

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=str, default="dB", help="power or dB")

    # BASIS hyperparameters
    parser.add_argument("--T", type=int, default=100,
                        help="Number of iteration in the inner loop")

    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigmaL', type=float, default=0.01)
    parser.add_argument('--num_classes', type=float, default=10)
    parser.add_argument('--progression', type=str, default='geometric')

    # Model hyperparameters
    parser.add_argument('--n_filters', type=int, default=192,
                        help="number of filters in the Network")

    # Glow hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")

    # Optimization parameters
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # preprocessing parameters
    parser.add_argument('--use_logit', action="store_true",
                        help="Either to use logit function to preprocess the data")
    parser.add_argument('--alpha', type=float, default=10**(-6),
                        help='preprocessing parameter: x = logit(alpha + (1 - alpha) * z / 256.). Only if use logit')

    args = parser.parse_args()

    main(args)
