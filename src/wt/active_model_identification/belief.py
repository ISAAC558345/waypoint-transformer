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


def quadrature(belief: Belief, nodes: int) -> list[tuple[float, float]]:
    if nodes < 3:
        raise ValueError("quadrature nodes")
    width = (belief.hi - belief.lo) / nodes
    points = [belief.lo + (index + 0.5) * width for index in range(nodes)]
    weights = [belief.kernel_weight(theta) for theta in points]
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("quadrature weight")
    return [(theta, weight / total) for theta, weight in zip(points, weights)]


def telemetry_y(true_drop_j: float, aux_j: float, epsilon_j: float) -> float:
    """Meter noise hits the observation only. The energy ledger stays noiseless."""
    meter = true_drop_j + epsilon_j
    return meter - aux_j
