"""Optical Bloch equations for a driven three-level emitter in dynamiqs.

The quantum counterpart of Part A.2. Operators are plain 3x3 matrices built from
``dq.basis(3, i)`` outer products (states |a>, |b>, |c> at indices 0, 1, 2). The
simulation runs in the frame rotating at the drive frequency omega, so it
produces the *rotating-frame* coherence rho_ba directly, comparable term by term
to ``ThreeLevelEmitter.steady_state_coherence`` with no extra phase bookkeeping.

H_RWA = -Delta |b><b| + (Omega0/2)(|b><a| + |a><b|),  Delta = omega - omega_ab.
Jump operators: sqrt(Gamma_ba)|a><b|, sqrt(Gamma_bc)|c><b|, sqrt(Gamma_ca)|a><c|,
sqrt(gamma_phi/2)(|b><b| - |a><a|).

``dq.set_precision('double')`` is called on import: dynamiqs/JAX default to
float32, which would degrade every steady-state comparison by ~3 orders of
magnitude (Part A ground rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp
from jax import Array

from qubit_playground.light_matter.emitter import ThreeLevelEmitter

dq.set_precision("double")

# Basis kets |a>, |b>, |c> and the projectors / transition operators.
_KET_A = dq.basis(3, 0)
_KET_B = dq.basis(3, 1)
_KET_C = dq.basis(3, 2)
_SIGMA_MINUS = _KET_A @ dq.dag(_KET_B)  # |a><b|; <sigma_-> = rho_ba
_PROJ_A = _KET_A @ dq.dag(_KET_A)
_PROJ_B = _KET_B @ dq.dag(_KET_B)
_PROJ_C = _KET_C @ dq.dag(_KET_C)


@dataclass(frozen=True)
class BlochResult:
    """Outcome of a driven-emitter master-equation simulation.

    Attributes:
        times: Save times, shape ``(n_times,)`` (fs).
        populations: Real populations (rho_aa, rho_bb, rho_cc), shape
            ``(3, n_times)``.
        coherence_ba: Complex rotating-frame coherence rho_ba, shape
            ``(n_times,)``.
        emitter: The emitter that was simulated.
        omega_drive: Drive angular frequency (1/fs).
        rabi: Full Rabi amplitude Omega0 (1/fs).
    """

    times: Array
    populations: Array
    coherence_ba: Array
    emitter: ThreeLevelEmitter
    omega_drive: float
    rabi: float


def _build_hamiltonian(emitter: ThreeLevelEmitter, omega_drive: float, rabi: float):
    """H_RWA in the drive-rotating frame (Part A.2)."""
    delta = omega_drive - emitter.omega_ab  # Delta = omega - omega_ab
    return -delta * _PROJ_B + (rabi / 2.0) * (_SIGMA_MINUS + dq.dag(_SIGMA_MINUS))


def _build_jump_ops(emitter: ThreeLevelEmitter) -> list:
    """Lindblad jump operators (Part A.2); zero-rate channels are skipped."""
    jumps = []
    if emitter.gamma_ba > 0.0:
        jumps.append(jnp.sqrt(emitter.gamma_ba) * (_KET_A @ dq.dag(_KET_B)))
    if emitter.gamma_bc > 0.0:
        jumps.append(jnp.sqrt(emitter.gamma_bc) * (_KET_C @ dq.dag(_KET_B)))
    if emitter.gamma_ca > 0.0:
        jumps.append(jnp.sqrt(emitter.gamma_ca) * (_KET_A @ dq.dag(_KET_C)))
    if emitter.gamma_phi > 0.0:
        # sqrt(gamma_phi/2)(|b><b| - |a><a|) adds exactly gamma_phi to rho_ba decay.
        jumps.append(jnp.sqrt(emitter.gamma_phi / 2.0) * (_PROJ_B - _PROJ_A))
    return jumps


def simulate_driven_emitter(
    emitter: ThreeLevelEmitter,
    *,
    omega_drive: float,
    rabi: float,
    t_final: float,
    n_times: int = 400,
    initial_state: Array | None = None,
) -> BlochResult:
    """Simulate the driven three-level emitter with dynamiqs.

    Args:
        emitter: The emitter to simulate.
        omega_drive: Drive angular frequency omega (1/fs).
        rabi: Full Rabi amplitude Omega0 (1/fs).
        t_final: Final simulation time (fs).
        n_times: Number of save points on [0, t_final].
        initial_state: Initial ket or density matrix. Defaults to ground |a>.

    Returns:
        A BlochResult with populations, coherence, and the inputs.
    """
    hamiltonian = _build_hamiltonian(emitter, omega_drive, rabi)
    jump_ops = _build_jump_ops(emitter)
    rho0 = _KET_A if initial_state is None else initial_state
    tsave = jnp.linspace(0.0, t_final, n_times)

    result = dq.mesolve(
        hamiltonian,
        jump_ops,
        rho0,
        tsave,
        exp_ops=[_PROJ_A, _PROJ_B, _PROJ_C, _SIGMA_MINUS],
        # Tight tolerances: weak-drive steady-state populations reach ~1e-4, so
        # the default atol=1e-6 would floor the accuracy of every comparison.
        method=dq.method.Tsit5(rtol=1e-10, atol=1e-12),
    )

    populations = jnp.real(result.expects[:3])
    coherence_ba = result.expects[3]
    return BlochResult(
        times=tsave,
        populations=populations,
        coherence_ba=coherence_ba,
        emitter=emitter,
        omega_drive=omega_drive,
        rabi=rabi,
    )


def steady_state(
    emitter: ThreeLevelEmitter,
    *,
    omega_drive: float,
    rabi: float,
    n_lifetimes: float = 30.0,
) -> BlochResult:
    """Evolve long enough to reach steady state, asserting convergence.

    Uses ``t_final = n_lifetimes / slowest_rate`` where ``slowest_rate`` is the
    minimum over the *nonzero* population decay rates. Convergence is checked by
    comparing the last two save points on every observable (atol 1e-8); if that
    trips, the system genuinely has not converged and returning would poison
    downstream comparisons. Explicit time evolution is used deliberately (no
    ``dq.steadystate`` dependency); the systems are 3x3.

    Args:
        emitter: The emitter to simulate.
        omega_drive: Drive angular frequency omega (1/fs).
        rabi: Full Rabi amplitude Omega0 (1/fs).
        n_lifetimes: Number of slowest lifetimes to evolve for.

    Returns:
        A BlochResult at the converged final time.
    """
    decay_rates = [
        r for r in (emitter.gamma_ba, emitter.gamma_bc, emitter.gamma_ca) if r > 0.0
    ]
    slowest_rate = min(decay_rates)
    t_final = n_lifetimes / slowest_rate

    result = simulate_driven_emitter(
        emitter,
        omega_drive=omega_drive,
        rabi=rabi,
        t_final=t_final,
    )

    last_pop = result.populations[:, -1]
    prev_pop = result.populations[:, -2]
    last_coh = result.coherence_ba[-1]
    prev_coh = result.coherence_ba[-2]
    converged = bool(
        jnp.all(jnp.abs(last_pop - prev_pop) < 1e-8)
        and jnp.abs(last_coh - prev_coh) < 1e-8
    )
    if not converged:
        raise RuntimeError(
            "steady_state did not converge; increase n_lifetimes "
            "(last two save points still differ by > 1e-8)"
        )
    return result
