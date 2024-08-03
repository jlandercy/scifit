import numpy as np
from scipy import special, stats


def voigt(x, sigma, gamma, x0=0., A=1.):
    """
    Shifted and Scaled Voigt Profile

    :param x:
    :param sigma:
    :param gamma:
    :param x0:
    :param A:
    :return:
    """
    z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
    y = special.erfcx(-1j * z) / (sigma * np.sqrt(2 * np.pi))
    return A * np.real(y)


def pseudo_voigt(x, eta, sigma, gamma, x0=0., A=1.):
    """
    Shifted and scaled Pseudo Voigt Profile

    :param x:
    :param eta:
    :param sigma:
    :param gamma:
    :param x0:
    :param A:
    :return:
    """
    G = stats.norm.pdf(x, scale=sigma, loc=x0)
    L = stats.cauchy.pdf(x, scale=2. * gamma, loc=x0)
    return A * ((1. - eta) * G + eta * L)
