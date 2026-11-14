"""Truncated-Gaussian propulsion belief.

The kernel center and precision are the conjugate parameters. Reported
expectations and quantiles are those of the distribution after the positive
support restriction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass(frozen=True)
class Belief:
    mu: float
    lam: float
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lam <= 0.0 or self.hi <= self.lo:
            raise ValueError("belief support")

    def copy(self) -> "Belief":
        return Belief(self.mu, self.lam, self.lo, self.hi)

    @property
    def kernel_std(self) -> float:
        return 1.0 / math.sqrt(self.lam)

    def _limits(self) -> tuple[float, float, float]:
        sigma = self.kernel_std
        return (self.lo - self.mu) / sigma, (self.hi - self.mu) / sigma, sigma

    def _mass(self) -> tuple[float, float, float, float]:
        alpha, beta, sigma = self._limits()
        mass = normal_cdf(beta) - normal_cdf(alpha)
        if mass < 1e-18:
            raise ValueError("belief mass vanished")
        return alpha, beta, sigma, mass

    def moments(self) -> tuple[float, float]:
        alpha, beta, sigma, mass = self._mass()
        gap = normal_pdf(alpha) - normal_pdf(beta)
        mean = self.mu + sigma * gap / mass
        second = 1.0 + (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / mass - (gap / mass) ** 2
        variance = max(0.0, sigma * sigma * second)
        return mean, variance

    def mean(self) -> float:
        return self.moments()[0]

    def variance(self) -> float:
        return self.moments()[1]

    def quantile(self, probability: float) -> float:
        if not 0.0 < probability < 1.0:
            raise ValueError("probability")
        alpha, beta, sigma, mass = self._mass()
        target = normal_cdf(alpha) + probability * mass
        left, right = self.lo, self.hi
        for _ in range(80):
            mid = 0.5 * (left + right)
            if normal_cdf((mid - self.mu) / sigma) < target:
                left = mid
            else:
                right = mid
        return 0.5 * (left + right)

    def update(self, z_j: float, y_j: float, sigma_j: float) -> "Belief":
        if sigma_j <= 0.0:
            raise ValueError("observation noise")
        if z_j == 0.0:
            return self.copy()
        precision = self.lam + (z_j * z_j) / (sigma_j * sigma_j)
        center = (self.lam * self.mu + z_j * y_j / (sigma_j * sigma_j)) / precision
        return Belief(center, precision, self.lo, self.hi)

    def kernel_weight(self, theta: float) -> float:
        if theta < self.lo or theta > self.hi:
            return 0.0
        delta = theta - self.mu
        return math.exp(-0.5 * self.lam * delta * delta)


def prior_belief(spec: dict) -> Belief:
    block = spec["belief"]
    lo, hi = float(block["support"][0]), float(block["support"][1])
    std = float(block["prior_kernel_std"])
    return Belief(float(block["prior_mu"]), 1.0 / (std * std), lo, hi)


def fixed_model_belief(spec: dict) -> Belief:
    block = spec["belief"]
    lo, hi = float(block["support"][0]), float(block["support"][1])
    std = float(block["fixed_model_kernel_std"])
    return Belief(float(block["fixed_model_mu"]), 1.0 / (std * std), lo, hi)


def equal_width_quadrature(belief: Belief, nodes: int) -> list[tuple[float, float]]:
    """Support-wide midpoints. A narrow kernel leaves most nodes in the tails.

    For a kernel centered at 1.25 with standard deviation 0.02 on [0.8, 1.4],
    seven midpoints renormalize to a mean near 1.2706 instead of the truncated
    mean. Callers that need the expectation use ``quadrature``.
    """
    if nodes < 3:
        raise ValueError("quadrature nodes")
    width = (belief.hi - belief.lo) / nodes
    points = [belief.lo + (index + 0.5) * width for index in range(nodes)]
    weights = [belief.kernel_weight(theta) for theta in points]
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("quadrature weight")
    return [(theta, weight / total) for theta, weight in zip(points, weights)]


_GL_UNIT: dict[int, tuple[tuple[float, float], ...]] = {}


def gauss_legendre_unit(nodes: int) -> tuple[tuple[float, float], ...]:
    """Gauss-Legendre nodes and weights for the integral over (0, 1)."""
    cached = _GL_UNIT.get(nodes)
    if cached is not None:
        return cached
    if nodes < 1:
        raise ValueError("quadrature nodes")
    found: list[tuple[float, float]] = []
    for index in range(nodes):
        x = math.cos(math.pi * (index + 0.75) / (nodes + 0.5))
        for _ in range(80):
            poly_nm1 = 1.0
            poly_n = x
            for degree in range(2, nodes + 1):
                poly_nm1, poly_n = poly_n, ((2 * degree - 1) * x * poly_n - (degree - 1) * poly_nm1) / degree
            derivative = nodes * (poly_nm1 - x * poly_n) / (1.0 - x * x)
            step = poly_n / derivative
            x -= step
            if abs(step) < 1e-15:
                break
        poly_nm1 = 1.0
        poly_n = x
        for degree in range(2, nodes + 1):
            poly_nm1, poly_n = poly_n, ((2 * degree - 1) * x * poly_n - (degree - 1) * poly_nm1) / degree
        derivative = nodes * (poly_nm1 - x * poly_n) / (1.0 - x * x)
        weight = 2.0 / ((1.0 - x * x) * derivative * derivative)
        found.append(((x + 1.0) / 2.0, weight / 2.0))
    found.sort(key=lambda item: item[0])
    cached = tuple(found)
    _GL_UNIT[nodes] = cached
    return cached


def quadrature(belief: Belief, nodes: int) -> list[tuple[float, float]]:
    """Expectation nodes on the quantile coordinate.

    The integral of f over the belief is sum w_i f(F^{-1}(u_i)), with
    Gauss-Legendre nodes u_i in (0, 1). Equal-width support nodes stay in
    ``equal_width_quadrature`` as the known narrow-posterior counterexample.
    """
    if nodes < 3:
        raise ValueError("quadrature nodes")
    return [(belief.quantile(probability), weight) for probability, weight in gauss_legendre_unit(nodes)]


def expectation(belief: Belief, nodes: int, function) -> float:
    return math.fsum(weight * float(function(theta)) for theta, weight in quadrature(belief, nodes))


# Physicist Gauss-Hermite nodes for weight exp(-x^2), n=5.
# Standard-normal nodes are sqrt(2) * x and weights are w / sqrt(pi).
_PHYSICIST_HERMITE_5 = (
    (0.0, 0.9453087204829419),
    (0.9585724646138185, 0.39361932315224115),
    (-0.9585724646138185, 0.39361932315224115),
    (2.0201828704560856, 0.019953242059045913),
    (-2.0201828704560856, 0.019953242059045913),
)


def gaussian_quadrature(sigma: float, nodes: int) -> list[tuple[float, float]]:
    """Nodes for epsilon ~ N(0, sigma^2). n=3 and n=5 are Gauss-Hermite.

    A one-node rule is the zero-noise mean observation. It is not the
    expectation of a nonlinear function of the future measurement.
    """
    if sigma <= 0.0:
        raise ValueError("noise scale")
    if nodes == 1:
        return [(0.0, 1.0)]
    if nodes == 3:
        scale = math.sqrt(3.0) * sigma
        return [(-scale, 1.0 / 6.0), (0.0, 2.0 / 3.0), (scale, 1.0 / 6.0)]
    if nodes == 5:
        root = math.sqrt(2.0) * sigma
        scale = math.sqrt(math.pi)
        return [(root * x, weight / scale) for x, weight in _PHYSICIST_HERMITE_5]
    if nodes < 3:
        raise ValueError("noise nodes")
    weight = 1.0 / nodes
    points = []
    for index in range(nodes):
        probability = (index + 0.5) * weight
        points.append(sigma * _standard_quantile(probability))
    return [(point, weight) for point in points]


def _standard_quantile(probability: float) -> float:
    left, right = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (left + right)
        if normal_cdf(mid) < probability:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def telemetry_y(true_drop_j: float, aux_j: float, epsilon_j: float) -> float:
    """Meter noise hits the observation only. The energy ledger stays noiseless."""
    meter = true_drop_j + epsilon_j
    return meter - aux_j
