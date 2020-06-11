import tensorflow as tf


@tf.function
def compute_grad_logprob(X, model):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = -tf.reduce_mean(model.log_prob(X))
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

def basis_inner_loop(mixed, x1, x2, model1, model2, sigma, n_mixed, sigmaL=0.01, delta=2e-5, T=100, dataset="mnist"):

    if dataset == 'mnist':
        data_shape = [n_mixed, 32, 32, 1]
    elif dataset == 'cifar10':
        data_shape == [n_mixed, 32, 32, 3]

    eta = delta * (sigma / sigmaL) ** 2
    lambda_recon = 1.0 / (sigma ** 2)
    for t in range(T):

        epsilon1 = tf.math.sqrt(2 * eta) * tf.random.normal(data_shape)
        epsilon2 = tf.math.sqrt(2 * eta) * tf.random.normal(data_shape)

        grad_logprob1 = compute_grad_logprob(x1, model1)
        grad_logprob2 = compute_grad_logprob(x2, model2)
        x1 = x1 + eta * (grad_logprob1 - lambda_recon * (x1 + x2 - 2 * mixed)) + epsilon1
        x2 = x2 + eta * (grad_logprob2 - lambda_recon * (x1 + x2 - 2 * mixed)) + epsilon2

    return x1, x2
