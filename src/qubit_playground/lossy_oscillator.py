"""Lossy harmonic oscillator: the canonical first open-quantum-system example.

A single cavity mode with Hamiltonian ``H = omega * a.dag() @ a`` under
single-photon loss (jump operator ``sqrt(kappa) * a``) obeys the Lindblad master
equation. Starting from a coherent state ``|alpha_0>`` the mean photon number
decays purely exponentially,

    <a.dag() a>(t) = |alpha_0|**2 * exp(-kappa * t),

because the Hamiltonian commutes with the number operator and only the
dissipator removes photons. This module simulates that decay with dynamiqs and
exposes the analytic curve alongside it so the two can be compared in a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class DecayResult:
    """Outcome of a lossy-oscillator simulation.

    Attributes:
        times: Sample times at which the state was saved, shape ``(n_times,)``.
        photon_number: Simulated mean photon number ``<a.dag() a>`` at each time.
        analytic: Analytic mean photon number ``|alpha_0|**2 * exp(-kappa t)``.
        kappa: Single-photon loss rate used for the simulation.
    """

    times: Array
    photon_number: Array
    analytic: Array
    kappa: float


def simulate_photon_decay(
    *,
    dim: int = 16,
    omega: float = 1.0,
    kappa: float = 0.2,
    alpha_0: float = 2.0,
    t_final: float = 20.0,
    n_times: int = 200,
) -> DecayResult:
    """Simulate photon-number decay of a lossy harmonic oscillator.

    Args:
        dim: Fock-space truncation (number of basis states).
        omega: Oscillator angular frequency (coefficient of ``a.dag() @ a``).
        kappa: Single-photon loss rate; the jump operator is ``sqrt(kappa) * a``.
        alpha_0: Amplitude of the initial coherent state ``|alpha_0>``.
        t_final: Final simulation time.
        n_times: Number of save points on ``[0, t_final]``.

    Returns:
        A :class:`DecayResult` holding the save times, the simulated mean photon
        number, the analytic reference curve, and the loss rate.
    """
    a = dq.destroy(dim)
    hamiltonian = omega * dq.dag(a) @ a
    jump_ops = [jnp.sqrt(kappa) * a]
    rho0 = dq.coherent(dim, alpha_0)
    tsave = jnp.linspace(0.0, t_final, n_times)

    result = dq.mesolve(
        hamiltonian,
        jump_ops,
        rho0,
        tsave,
        exp_ops=[dq.number(dim)],
    )

    photon_number = jnp.real(result.expects[0])
    analytic = alpha_0**2 * jnp.exp(-kappa * tsave)

    return DecayResult(
        times=tsave,
        photon_number=photon_number,
        analytic=analytic,
        kappa=kappa,
    )


def max_absolute_error(result: DecayResult) -> float:
    """Return the largest absolute deviation from the analytic decay curve.

    Args:
        result: A simulation outcome produced by :func:`simulate_photon_decay`.

    Returns:
        The maximum absolute difference between the simulated and analytic mean
        photon number over all save times.
    """
    return float(jnp.max(jnp.abs(result.photon_number - result.analytic)))
