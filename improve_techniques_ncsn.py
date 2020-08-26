import tensorflow as tf
import argparse
from pipeline import data_loader

def main(args):

    ds_train, _, _, _, _ = data_loader.load_melspec_ds(args.dataset + '/train', args.dataset + '/test',
                                                       shuffle=True, batch_size=None, mirrored_strategy=None)

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

    def map_fn(X):
        X = (X - args.minval) / (args.maxval - args.minval)
        return X

    train_dataset = ds_train.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    train_list = (list(train_dataset.as_numpy_iterator()))

    print("Number of spectrograms in training set: {}".format(len(train_list)))

    max_euclidean_distances = 0.
    for i in range(len(train_list)):
        for j in range(i + 1, len(train_list)):
            if i != j:
                euclidean_distance = tf.norm(train_list[i] - train_list[j], ord='euclidean')
                if euclidean_distance > max_euclidean_distances:
                    max_euclidean_distances = euclidean_distance

        if i % (len(train_list) // 10) == 0:
            print("Finish Step {}. Current max: {}".format(i, max_euclidean_distances))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute Sigma1 for NCSNv2')

    parser.add_argument('dataset', type=str, help='dirpath of the dataset')
