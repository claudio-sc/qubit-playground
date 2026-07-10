"""Weak coupling / Purcell effect end-to-end (paper Sec. 4B, spec Step 5).

Demonstrates and tests the chain
    classical S(omega_0) -> (gamma_p, delta_omega) -> quantum simulation
    -> extracted eigenfrequency.

Honesty note (mirrored in the README): in weak coupling the quantum simulation
*consumes* classically computed rates -- the physics content is the bridge
normalization S_tilde = 2*Gamma_0*S; the simulation demonstrates the modified
dynamics rather than independently predicting it. The independent cross-check is
Step 6 (strong coupling).

Sign convention: the dressed transition frequency is omega_p = omega_0 +
delta_omega, so in the frame rotating at the bare omega_0 the coherence evolves
as <sigma_-(t)> ~ exp(-i*delta_omega*t - gamma_p*t/2) and the matrix-pencil pole
must be exactly ``pole = delta_omega - 1j*gamma_p/2``.

This module builds on dynamiqs (via ``bloch``); it is one of the three modules
allowed to touch the quantum stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import dynamiqs as dq
import numpy as np
from jax import Array

from qubit_playground.light_matter.bloch import simulate_driven_emitter
from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import EnvironmentSpectrum
from qubit_playground.light_matter.fitting import fit_complex_exponentials

dq.set_precision("double")


@dataclass(frozen=True)
class PurcellResult:
    """Outcome of a Purcell (weak-coupling) simulation.

    Attributes:
        times: Save times (fs).
        populations: Real populations (rho_aa, rho_bb, rho_cc), shape (3, n).
        coherence_ba: Complex rotating-frame coherence rho_ba, shape (n,).
        gamma_p: Environment-modified decay rate consumed by the simulation.
        delta_omega: Environment-induced frequency shift consumed.
        extracted_pole: Matrix-pencil pole of the coherence (should equal
            ``delta_omega - 1j*gamma_p/2``).
    """

    times: Array
    populations: Array
    coherence_ba: Array
    gamma_p: float
    delta_omega: float
    extracted_pole: complex


def simulate_modified_decay(
    spectrum: EnvironmentSpectrum,
    emitter: ThreeLevelEmitter,
    *,
    t_final: float,
    n_times: int = 400,
) -> PurcellResult:
    """Simulate the environment-dressed two-level decay (paper Sec. 4B).

    Computes ``gamma_p`` and ``delta_omega`` classically at omega_0 =
    ``emitter.omega_ab``, then evolves the dressed two-level system in the frame
    rotating at the bare omega_0: H = delta_omega |b><b|, jump sqrt(gamma_p)
    |a><b|, initial state (|a>+|b>)/sqrt(2).

    The dressed system is realised by reusing ``bloch.simulate_driven_emitter``
    with a two-level emitter whose gamma_ba = gamma_p, zero drive, and a drive
    frequency chosen so that the RWA detuning term ``-Delta |b><b|`` becomes
    ``delta_omega |b><b|`` (Delta = -delta_omega).

    Args:
        spectrum: Classical environment spectrum for this position.
        emitter: A two-level emitter; its omega_ab sets omega_0.
        t_final: Final simulation time (fs).
        n_times: Number of save points.

    Returns:
        A PurcellResult with the consumed rates and the extracted pole.
    """
    omega_0 = emitter.omega_ab
    gamma_p = spectrum.decay_rate(emitter.gamma_ba, emitter.gamma_coherence, omega_0)
    delta_omega = spectrum.frequency_shift(emitter.gamma_ba, omega_0)

    # Dressed two-level system: coherence decays at gamma_p/2, no extra dephasing.
    dressed = ThreeLevelEmitter.two_level(omega_ab=omega_0, gamma_ba=gamma_p)
    psi0 = dq.unit(dq.basis(3, 0) + dq.basis(3, 1))
    # H_RWA = -Delta|b><b| with Delta = omega_drive - omega_0; choose
    # omega_drive = omega_0 - delta_omega so that H = delta_omega|b><b|.
    result = simulate_driven_emitter(
        dressed,
        omega_drive=omega_0 - delta_omega,
        rabi=0.0,
        t_final=t_final,
        n_times=n_times,
        initial_state=psi0,
    )

    coherence = np.asarray(result.coherence_ba)
    dt = float(result.times[1] - result.times[0])
    pole = complex(fit_complex_exponentials(coherence, dt, n_modes=1)[0])

    return PurcellResult(
        times=result.times,
        populations=result.populations,
        coherence_ba=result.coherence_ba,
        gamma_p=gamma_p,
        delta_omega=delta_omega,
        extracted_pole=pole,
    )
