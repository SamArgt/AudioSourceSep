import numpy as np
import functools
import itertools
from scipy.signal import stft, istft

"""
Oracle Systems for Source Separation

The code is taken from the https://github.com/sigsep/sigsep-mus-oracle and modified to take as inputs numpy arrays
"""


def IBM(mixture, sources, alpha=1, theta=0.5):
    """Ideal Binary Mask:
    processing all channels inpependently with the ideal binary mask.

    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as
    magnitude of STFT raised to the power alpha. Typical parameters involve a
    ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)

    Parameters
    ----
    mixture: np.ndarray, shape=(nsampl, nchan)
        matrix containing mixture

    sources: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing true sources

    alpha: ratio of matgnitudes for the spectrograms

    theta: majority vote


    Returns
    ----
    estimates: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing estimated sources
    """

    # parameters for STFT
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = mixture.shape[0]  # remember number of samples for future use
    X = stft(mixture.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # perform separation
    estimates = np.zeros_like(sources)
    for i, source in enumerate(list(sources)):

        # compute STFT of target source
        Yj = stft(source.T, nperseg=nfft)[-1]

        # Create Binary Mask
        Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
        Mask[np.where(Mask >= theta)] = 1
        Mask[np.where(Mask < theta)] = 0

        # multiply mask
        Yj = np.multiply(X, Mask)

        # inverte to time domain and set same length as original mixture
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[i, :] = target_estimate

    return estimates


def IRM(mixture, sources, alpha=2):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)

    Parameters
    ----
    mixture: np.ndarray, shape=(nsampl, nchan)
        matrix containing mixture

    sources: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing true sources

    alpha: ratio of matgnitudes for the spectrograms


    Returns
    ----
    estimates: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing estimated sources
    """

    # STFT parameters
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = mixture.shape[0]  # remember number of samples for future use
    X = stft(mixture.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = []
    # compute model as the sum of spectrograms
    model = eps

    for i, source in enumerate(list(sources)):
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        spec = np.abs(stft(source.audio.T, nperseg=nfft)[-1])**alpha
        model += spec
        P.append(spec)

    # now performs separation
    estimates = np.zeros_like(sources)
    for i, source in enumerate(list(sources)):
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[i]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[i, :] = target_estimate

    return estimates


def invert(M, eps):
    """"inverting matrices M (matrices are the two last dimensions).
    This is assuming that these are 2x2 matrices, using the explicit
    inversion formula available in that case."""
    invDet = 1.0 / (eps + M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0])
    invM = np.zeros(M.shape, dtype='complex')
    invM[..., 0, 0] = invDet * M[..., 1, 1]
    invM[..., 1, 0] = -invDet * M[..., 1, 0]
    invM[..., 0, 1] = -invDet * M[..., 0, 1]
    invM[..., 1, 1] = invDet * M[..., 0, 0]
    return invM


def MWF(mixture, sources):
    """Multichannel Wiener Filter:
    processing all channels jointly with the ideal multichannel filter
    based on the local gaussian model, assuming time invariant spatial
    covariance matrix.

    Parameters
    ----
    mixture: np.ndarray, shape=(nsampl, nchan)
        matrix containing mixture

    sources: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing true sources


    Returns
    ----
    estimates: np.ndarray, shape=(nsrc, nsampl, nchan)
        matric containing estimated sources
    """

    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # parameters for STFT
    nfft = 2048

    # compute STFT of Mixture
    N = mixture.shape[0]  # remember number of samples for future use
    X = stft(mixture.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Allocate variables P: PSD, R: Spatial Covarianc Matrices
    P = []
    R = []
    for i, source in enumerate(list(sources)):

        # compute STFT of target source
        Yj = stft(source.T, nperseg=nfft)[-1]

        # Learn Power Spectral Density and spatial covariance matrix
        # -----------------------------------------------------------

        # 1/ compute observed covariance for source
        Rjj = np.zeros((F, T, I, I), dtype='complex')
        for (i1, i2) in itertools.product(range(I), range(I)):
            Rjj[..., i1, i2] = Yj[i1, ...] * np.conj(Yj[i2, ...])

        # 2/ compute first naive estimate of the source spectrogram as the
        #    average of spectrogram over channels
        P.append(np.mean(np.abs(Yj)**2, axis=0))

        # 3/ take the spatial covariance matrix as the average of
        #    the observed Rjj weighted Rjj by 1/Pj. This is because the
        #    covariance is modeled as Pj Rj
        R.append(np.mean(Rjj / (eps + P[i][..., None, None]), axis=1))

        # add some regularization to this estimate: normalize and add small
        # identify matrix, so we are sure it behaves well numerically.
        R[i] = R[i] * I / np.trace(R[i]) + eps * np.tile(
            np.eye(I, dtype='complex64')[None, ...], (F, 1, 1)
        )

        # 4/ Now refine the power spectral density estimate. This is to better
        #    estimate the PSD in case the source has some correlations between
        #    channels.

        #    invert Rj
        Rj_inv = invert(R[i], eps)

        #    now compute the PSD
        P[i] = 0
        for (i1, i2) in itertools.product(range(I), range(I)):
            P[i] += 1. / I * np.real(
                Rj_inv[:, i1, i2][:, None] * Rjj[..., i2, i1]
            )

    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    Cxx = 0
    for i, source in enumerate(list(sources)):
        Cxx += P[i][..., None, None] * R[i][:, None, ...]

    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx, eps)

    # now separate sources
    estimates = np.zeros_like(sources)
    for i, source in enumerate(list(sources)):
        # computes multichannel Wiener gain as Pj Rj invCxx
        G = np.zeros(invCxx.shape, dtype='complex64')
        SR = P[i][..., None, None] * R[i][:, None, ...]
        for (i1, i2, i3) in itertools.product(range(I), range(I), range(I)):
            G[..., i1, i2] += SR[..., i1, i3] * invCxx[..., i3, i2]
        SR = 0  # free memory

        # separates by (matrix-)multiplying this gain with the mix.
        Yj = 0
        for i in range(I):
            Yj += G[..., i] * X[i, ..., None]
        Yj = np.rollaxis(Yj, -1)  # gets channels back in first position

        # inverte to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[i, :] = target_estimate

    return estimates


def IBM_melspec(mixture, sources, theta=0.5):
    """Ideal Binary Mask:
    processing all channels inpependently with the ideal binary mask.

    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as
    magnitude of STFT raised to the power alpha. Typical parameters involve a
    ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)

    Parameters
    ----
    mixture: np.ndarray, shape=(nsample, f, t)
        matrix containing melspectrograms of the mixture

    sources: np.ndarray, shape=(nsrc, nsample, f, t)
        metrix containing melspectograms if the true sources

    theta: majority vote


    Returns
    ----
    estimates: np.ndarray, shape=(nsrc, f, t)
        matric containing estimated sources
    """

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # perform separation
    estimates = np.zeros_like(sources)
    for i, source in enumerate(list(sources)):

        # Create Binary Mask
        Mask = np.divide(source, (eps + mixture))
        Mask[np.where(Mask >= theta)] = 1
        Mask[np.where(Mask < theta)] = 0

        # multiply mask
        target_estimate = np.multiply(mixture, Mask)

        # set this as the source estimate
        estimates[i, :] = target_estimate

    return estimates


def IRM_melspec(mixture, sources, alpha=2):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)

    Parameters
    ----
    mixture: np.ndarray, shape=(nsample, f, t)
        matrix containing melspectrograms of the mixture

    sources: np.ndarray, shape=(nsrc, nsample, f, t)
        metrix containing melspectograms if the true sources

    theta: majority vote


    Returns
    ----
    estimates: np.ndarray, shape=(nsrc, f, t)
        matric containing estimated sources
    """
    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute model as the sum of spectrograms
    model = np.sum(sources, axis=0) + eps

    # now performs separation
    estimates = np.zeros_like(sources)
    for i, source in enumerate(list(sources)):
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(source, model)

        # multiply the mix by the mask
        target_estimate = np.multiply(mixture, Mask)

        estimates[i, :] = target_estimate

    return estimates
