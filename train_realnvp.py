import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import utils
from flow_models import flow_builder
from pipeline import data_loader
import argparse
import time
import os
import sys
from train_utils import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def main(args):
    event_shape = [28, 28, 1]
    flow = flow_builder.build_realnvp(event_shape, n_filters=32, n_blocks=4, learntop=True, mirrored_strategy=None)
    ds_train, ds_val, _ = data_loader.load_toydata(dataset='mnist', batch_size=256, preprocessing=False)
    optimizer = tfk.optimizers.Adam()
    print("flow sample shape: ", flow.sample(1).shape)

    total_trainable_variables = utils.total_trainable_variables(flow)
    print("Total Trainable Variables: ", total_trainable_variables)

    # The objective function to minimize is the negative log-likelihood
    def compute_loss(X):
        return tf.reduce_mean(-flow.log_prob(X))

    def train_step(X):
        with tf.GradientTape() as tape:
            tape.watch(flow.trainable_variables)
            loss = compute_loss(X)
        gradients = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(
            list(zip(gradients, flow.trainable_variables)))
        return loss

    def train(n_epochs):
        avg_train_loss = tfk.metrics.Mean()
        avg_test_loss = tfk.metrics.Mean()
        for epoch in range(n_epochs):
            avg_train_loss.reset_states()
            avg_test_loss.reset_states()
            for batch in ds_train:
                loss = train_step(batch)
                avg_train_loss.update_state(loss)

            if epoch % 5 == 0:
                for batch in ds_val:
                    loss = compute_loss(batch)
                    avg_test_loss.update_state(loss)
                print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                    epoch, avg_train_loss.result(), avg_test_loss.result()))
            else:
                print("Epoch {:03d}: Train Loss: {:.3f}".format(epoch, avg_train_loss.result()))

    print("Start Training on {} epochs".format(args.n_epochs))
    train(args.n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model')

    parser.add_argument("--n_epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
