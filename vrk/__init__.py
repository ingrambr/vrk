"""
vrk — Variable Rank Kriging.

Sparse Gaussian Process Kriging with Expectation Propagation,
using standard geostatistics covariance conventions.

EP site parameters follow the C++ convention:
    varEP[i] = -r / (1 + r * cavity_var)   (positive λ_i)
    meanEP[i] = cavity_mean - q/r           (ã_i)
"""

from vrk.core.vrk import VRK
from vrk.covariance.exponential import ExponentialCovariance
from vrk.covariance.spherical import SphericalCovariance
from vrk.covariance.gaussian import GaussianCovariance
from vrk.covariance.matern52 import Matern52Covariance
from vrk.covariance.nugget import NuggetCovariance
from vrk.covariance.sum import SumCovariance
from vrk.likelihoods.gaussian import GaussianLikelihood
from vrk.likelihoods.exponential_noise import ExponentialNoiseLikelihood
from vrk.likelihoods.bernoulli import BernoulliLikelihood
from vrk.likelihoods.student_t import StudentTLikelihood
from vrk.likelihoods.poisson import PoissonLikelihood
from vrk.likelihoods.gumbel import GumbelLikelihood

__all__ = [
    "VRK",
    "ExponentialCovariance", "SphericalCovariance", "GaussianCovariance",
    "Matern52Covariance", "NuggetCovariance", "SumCovariance",
    "GaussianLikelihood", "ExponentialNoiseLikelihood",
    "BernoulliLikelihood", "StudentTLikelihood", "PoissonLikelihood",
    "GumbelLikelihood",
]
