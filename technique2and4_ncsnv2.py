import numpy as np
from scipy import stats, optimize
import argparse

def technique2(D, sigma1, sigmaL):

    def t2(gamma):
        cdf1 = stats.norm.cdf(np.sqrt(2. * D) * (gamma - 1.) + 3. * gamma)
        cdf2 = stats.norm.cdf(np.sqrt(2. * D) * (gamma - 1.) - 3. * gamma)
        return cdf1 - cdf2 - 0.5

    opt = optimize.root_scalar(t2, x0=0.5, x1=1., bracket=[0.5, 1.])
    if opt.flag != 'converged':
        print("DID NOT FIND ROOT FOR GAMMA")

    gamma = opt.root
    # check gamma is root
    print('gamma={}'.format(round(gamma, 4)))
    print("C = t2(gamma) + 0.5 ={}".format(t2(gamma) + 0.5))

    sigma1 = sigma1
    sigmaL = sigmaL
    n = np.log(sigmaL / sigma1) / np.log(gamma)
    print('num_classes = {}'.format(round(n, 0)))

    return gamma


def technique4(T, sigmaL, gamma):
    def t4(epsilon):
        term1 = (1 - (epsilon / sigmaL**2)) ** (2 * T)
        term2 = gamma**2 - (2 * epsilon / (sigmaL**2 - (sigmaL**2) * (1 - epsilon / (sigmaL**2))**2))
        term3 = 2 * epsilon / (sigmaL**2 - (sigmaL**2) * (1 - epsilon / (sigmaL**2))**2)
        return term1 * term2 + term3 - 1

    opt = optimize.root_scalar(t4, x0=1e-6, x1=1e-4)
    if opt.flag != 'converged':
        print("DID NOT FIND ROOT FOR EPSILON")

    epsilon = opt.root
    # check gamma is root
    print('epsilon={}'.format(epsilon))
    print("1 = t4(epsilon) + 1 ={}".format(t4(epsilon) + 1.))


def main(args):

    assert args.sigma1 > args.sigmaL
    try:
        D = args.D.split(',')
        D = [int(i) for i in D]
        D = np.prod(D)
    except (ValueError, TypeError):
        print('ERROR: D should be in the form: H,W,C')
        return 1

    # Display parameters
    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n'.format(k, v)
    print(template)

    gamma = technique2(D, args.sigma1, args.sigmaL)
    technique4(args.T, args.sigmaL, gamma)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute num_classes and epsilon for NCSNv2')

    parser.add_argument('--D', type=str, help='Dimensions: H,W,C', default=[96, 64, 1])
    parser.add_argument('--T', type=float, help='number of step at each iteration in the Langevin Dynamics', default=5.)
    parser.add_argument('--sigma1', type=float, default=55.)
    parser.add_argument('--sigmaL', type=float, default=0.01)

    args = parser.parse_args()

    main(args)
