import numpy as np

from scipy import stats
from PyAnEn.dist_TruncatedGamma import truncgamma_gen


def test_gamma():

    x = np.linspace(-5, 15, 999)
    mu = 3.5
    sigma = 0.9
    shift = -2

    shape = (mu / sigma) ** 2
    scale = sigma ** 2 / mu

    # Test PDF
    y1 = stats.gamma(a=shape, scale=scale, loc=shift).pdf(x)
    y2 = truncgamma_gen()(a=(-shift)/scale, b=10000, s=shape, scale=scale, loc=shift).pdf(x)

    assert np.abs(np.sum(y1) * (x[2] - x[1]) - 1) < 1e-4
    assert np.abs(np.sum(y2) * (x[2] - x[1]) - 1) < 1e-4
    assert np.abs(np.sum(np.abs(y1 - y2)[x < 0]) - np.sum(np.abs(y1 - y2)[x > 0])) < 1e-3

    # Test CDF
    y1 = stats.gamma(a=shape, scale=scale, loc=shift).cdf(x)
    y2 = truncgamma_gen()(a=(-shift)/scale, b=10000, s=shape, scale=scale, loc=shift).cdf(x)

    assert np.abs(y1[-1] - 1) < 1e-5
    assert np.abs(y2[-1] - 1) < 1e-5

    # Test inverse
    a = truncgamma_gen()(a=(-shift)/scale, b=10000, s=shape, scale=scale, loc=shift).ppf(0.5)
    b = truncgamma_gen()(a=(-shift)/scale, b=10000, s=shape, scale=scale, loc=shift).cdf(a)

    assert np.abs(b - 0.5) < 1e-5

    # Test sampling
    samples = truncgamma_gen()(a=(-shift)/scale, b=10000, s=shape, scale=scale, loc=shift).rvs(size=[3, 4, 5])
    assert samples.shape == (3, 4, 5)
    assert samples.min() >= 0
