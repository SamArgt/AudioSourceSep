import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import flow_builder
from flow_models import utils
from pipeline import data_loader
import argparse
import time
import os
import sys
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def setUp_optimizer(args):
    lr = args.learning_rate
    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam(lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax(lr=lr)
    else:
        raise ValueError("optimizer argument should be adam or adamax")
    return optimizer


def restore_checkpoint(args_parsed, flow, optimizer):
    restore_abs_dirpath = os.path.abspath(args_parsed.RESTORE)
    if args_parsed.latest:
        checkpoint_restore_path = tf.train.latest_checkpoint(restore_abs_dirpath)
    else:
        checkpoint_restore_path = restore_abs_dirpath

    # Checkpoint object
    ckpt = tf.train.Checkpoint(
        variables=flow.variables, optimizer=optimizer)
    # Restore weights if specified
    status = ckpt.restore(checkpoint_restore_path)
    status.assert_existing_objects_matched()

    return ckpt


def evaluate(args_parsed, flow, ds, ds_val):

    D = tf.constant(28 * 28 * 1, dtype=tf.float32)  # dimension of the data

    @tf.function
    def eval_step(inputs):
        losses = flow.log_prob(inputs)

        log_lik = losses
        log_lik -= tf.math.log(256.) * D
        if args_parsed.use_logit:
            log_lik -= tf.reduce_sum(tf.math.sigmoid(inputs) * (1 - tf.math.sigmoid(inputs)), axis=[1, 2, 3])

        bits_per_pixel = - log_lik / (D * tf.math.log(2.))

        avg_loss = -tf.reduce_mean(losses)
        avg_neg_log_lik = -tf.reduce_mean(log_lik)
        avg_bits_per_pixel = tf.reduce_mean(bits_per_pixel)

        return avg_loss, avg_neg_log_lik, avg_bits_per_pixel

    train_loss = tfk.metrics.Mean(name='train loss')
    train_nll = tfk.metrics.Mean(name="train nll")
    train_bits_per_pixel = tfk.metrics.Mean(name="train bits per pixel")

    test_loss = tfk.metrics.Mean(name='test loss')
    test_nll = tfk.metrics.Mean(name="test nll")
    test_bits_per_pixel = tfk.metrics.Mean(name="test bits per pixel")

    for batch in ds:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch)

        train_loss.update_state(avg_loss)
        train_nll.update_state(avg_neg_log_lik)
        train_bits_per_pixel.update_state(avg_bits_per_pixel)

    for batch in ds_val:
        avg_loss, avg_neg_log_lik, avg_bits_per_pixel = eval_step(batch)

        test_loss.update_state(avg_loss)
        test_nll.update_state(avg_neg_log_lik)
        test_bits_per_pixel.update_state(avg_bits_per_pixel)

    return train_loss, train_nll, train_bits_per_pixel, test_loss, test_nll, test_bits_per_pixel


def main(args_parsed):

    result_file = open(args_parsed.output, "a")
    sys.stdout = result_file

    ds, ds_val, minibatch = data_loader.load_data(dataset=args_parsed.dataset, batch_size=args_parsed.batch_size,
                                                  use_logit=args_parsed.use_logit, alpha=args_parsed.alpha,
                                                  noise=args_parsed.noise, mirrored_strategy=None)

    flow = flow_builder.build_flow(minibatch, L=args_parsed.L, K=args_parsed.K, n_filters=args_parsed.n_filters,
                                   dataset=args_parsed.dataset, l2_reg=args_parsed.l2_reg,
                                   mirrored_strategy=None, learntop=args_parsed.learntop)

    optimizer = setUp_optimizer(args_parsed)

    restore_checkpoint(args_parsed, flow, optimizer)

    print('_' * 100)

    params_dict = vars(args_parsed)
    template = 'Glow Flow \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    print("Total Trainable Variables: ", utils.total_trainable_variables(flow))

    print("\nWeights restored from {} \n".format(args_parsed.RESTORE))

    print('Start Evaluation...\n')
    t0 = time.time()
    tfk_metrics = evaluate(args_parsed, flow, ds, ds_val)
    t1 = time.time()

    for m in tfk_metrics:
        print("{}: {}".format(m.name, m.result()))

    print('\nDuration: {} seconds'.format(round(t1 - t0, 3)))
    print('_' * 100)
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Flow model')
    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights')
    parser.add_argument('--latest', action="store_true",
                        help="Restore latest checkpoint from restore directory")
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='evaluation.txt',
                        help='output dirpath for savings')

    # Model hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=512,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")

    # Optimization parameters
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    # preprocessing parameters
    parser.add_argument('--use_logit', action="store_true",
                        help="Either to use logit function to preprocess the data")
    parser.add_argument('--alpha', type=float, default=10**(-6),
                        help='preprocessing parameter: x = logit(alpha + (1 - alpha) * z / 256.). Only if use logit')
    parser.add_argument('--noise', type=float, default=None, help='noise level for BASIS separation')

    args_parsed = parser.parse_args()

    main(args_parsed)
