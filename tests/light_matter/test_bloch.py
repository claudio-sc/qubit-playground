import dynamiqs as dq
import numpy as np
import pytest

from qubit_playground.light_matter.bloch import (
    BlochResult,
    simulate_driven_emitter,
    steady_state,
)
from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.fitting import fit_complex_exponentials

dq.set_precision("double")


@pytest.fixture(scope="module")
def undriven_decay() -> BlochResult:
    emitter = ThreeLevelEmitter(omega_ab=3.0, gamma_ba=0.1, gamma_bc=0.05, gamma_ca=0.2)
    total = emitter.gamma_ba + emitter.gamma_bc
    return simulate_driven_emitter(
        emitter,
        omega_drive=3.0,
        rabi=0.0,
        t_final=8.0 / total,
        n_times=400,
        initial_state=dq.basis(3, 1),  # start in |b>
    )


def test_undriven_decay_rate_and_branching(undriven_decay: BlochResult) -> None:
    emitter = undriven_decay.emitter
    total = emitter.gamma_ba + emitter.gamma_bc
    times = np.asarray(undriven_decay.times)
    rho_bb = np.asarray(undriven_decay.populations[1])
    slope, _ = np.polyfit(times, np.log(rho_bb), 1)
    assert np.isclose(slope, -total, rtol=1e-3)
    # Everything ends in |a>; |c> is transiently populated.
    assert undriven_decay.populations[0, -1] > 0.99
    assert float(np.max(undriven_decay.populations[2])) > 0.0


@pytest.fixture(scope="module")
def weak_two_level() -> ThreeLevelEmitter:
    return ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)


@pytest.mark.parametrize("delta_over_half_width", [0.0, 1.0])
def test_steady_state_weak_drive_matches_analytic(
    weak_two_level: ThreeLevelEmitter, delta_over_half_width: float
) -> None:
    # s << 1: analytic RWA steady state is exact and the simulation *is* the
    # RWA master equation, so agreement is limited only by ODE + truncation.
    emitter = weak_two_level
    rabi = 1e-2 * emitter.gamma_ba
    half_width = emitter.gamma_coherence / 2.0
    omega_drive = emitter.omega_ab - delta_over_half_width * half_width

    result = steady_state(emitter, omega_drive=omega_drive, rabi=rabi)
    _, rho_bb_a, _ = emitter.steady_state_populations(omega_drive, rabi)
    rho_ba_a = emitter.steady_state_coherence(omega_drive, rabi)

    assert np.isclose(float(result.populations[1, -1]), rho_bb_a, rtol=1e-4)
    assert np.isclose(complex(result.coherence_ba[-1]), rho_ba_a, rtol=1e-4)


def test_steady_state_saturated_matches_analytic() -> None:
    # s ~ 5, three-level with fast |b>->|c> and dephasing: this catches wrong
    # jump operators, since both reshape the saturation curve.
    emitter = ThreeLevelEmitter(
        omega_ab=3.0,
        gamma_ba=0.05,
        gamma_bc=0.5,
        gamma_ca=0.05,
        gamma_phi=0.02,
    )
    rabi = 0.5
    omega_drive = emitter.omega_ab
    # Sanity: this is genuinely a saturated working point.
    assert emitter.saturation(omega_drive, rabi) > 1.0

    result = steady_state(emitter, omega_drive=omega_drive, rabi=rabi)
    rho_aa_a, rho_bb_a, rho_cc_a = emitter.steady_state_populations(omega_drive, rabi)
    rho_ba_a = emitter.steady_state_coherence(omega_drive, rabi)

    assert np.isclose(float(result.populations[0, -1]), rho_aa_a, rtol=1e-4)
    assert np.isclose(float(result.populations[1, -1]), rho_bb_a, rtol=1e-4)
    assert np.isclose(float(result.populations[2, -1]), rho_cc_a, rtol=1e-4)
    assert np.isclose(complex(result.coherence_ba[-1]), rho_ba_a, rtol=1e-4)


def test_coherence_damping_pole() -> None:
    # Undriven, on resonance (Delta=0 so no oscillation in the rotating frame),
    # initial (|a>+|b>)/sqrt(2). The single pole is varpi = -i*gamma_ab/2.
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1, gamma_phi=0.05)
    g = emitter.gamma_coherence
    psi0 = dq.unit(dq.basis(3, 0) + dq.basis(3, 1))
    result = simulate_driven_emitter(
        emitter,
        omega_drive=emitter.omega_ab,
        rabi=0.0,
        t_final=8.0 / g,
        n_times=400,
        initial_state=psi0,
    )
    coherence = np.asarray(result.coherence_ba)
    dt = float(result.times[1] - result.times[0])
    pole = fit_complex_exponentials(coherence, dt, n_modes=1)[0]
    assert abs(pole.real) < 1e-3 * g  # no oscillation
    assert np.isclose(pole.imag, -g / 2.0, rtol=1e-3)
