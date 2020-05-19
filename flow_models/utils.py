import numpy as np


def print_summary(flow):
    """
    Print a summary of the trainable varaibles of a flow model

    """

    print('_' * 100)
    print('Network \t Layer \t Shape \t\t\t\t\t\t #parameters')
    print('=' * 100)
    total_count = 0
    curr_network_name = ''
    curr_layer_name = ''
    curr_print_shape = ''
    curr_count = 0
    not_first_print = False

    for elt in flow.trainable_variables:
        name = elt.name
        names = name.split('/')
        kernel_bias = names[-1].split(':')[0]

        if len(names) == 1:
            network_name = ''
            layer_name = kernel_bias
        elif len(names) > 3:
            network_name = '/'.join([names[-4], names[-3]])
            layer_name = names[-2]
        elif len(names) >= 3:
            network_name = names[-3]
            layer_name = names[-2]
        else:
            layer_name = names[-2]
            network_name = ''

        shape = elt.shape
        count = np.prod(shape)

        if layer_name == curr_layer_name:
            curr_print_shape += ' | ' + kernel_bias + ' ' + str(shape)
            curr_count += count
        else:
            if total_count != 0:
                print('\t', curr_layer_name, '\t\t',
                      curr_print_shape, '\t\t', curr_count)
            curr_layer_name = layer_name
            curr_print_shape = kernel_bias + ' ' + str(shape)
            curr_count = count
            if (network_name != curr_network_name) or network_name == '':
                if not_first_print is True:
                    print('_' * 100)
                else:
                    not_first_print = True
                print(network_name)
                curr_network_name = network_name

        total_count += count

    print('\t', curr_layer_name, '\t\t', curr_print_shape, '\t\t', curr_count)
    print('=' * 100)
    print("Total Trainable Variables: ", total_count)
