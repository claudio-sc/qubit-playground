import numpy as np
import pytest

from qubit_playground.light_matter.emitter import ThreeLevelEmitter


def test_two_level_on_resonance_saturation_anchor() -> None:
    # Textbook anchor: s = 2*Omega0^2/Gamma_ba^2 on resonance, gamma_phi = 0.
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    rabi = 0.05
    s = emitter.saturation(omega_drive=3.0, rabi=rabi)
    assert np.isclose(s, 2.0 * rabi**2 / emitter.gamma_ba**2, rtol=1e-12)


def test_two_level_strong_drive_population_half() -> None:
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    rabi = 1e3 * emitter.gamma_ba
    _, rho_bb, _ = emitter.steady_state_populations(omega_drive=3.0, rabi=rabi)
    assert rho_bb > 0.49


def test_weak_drive_population_is_lorentzian_in_detuning() -> None:
    # At s ~ 0, rho_bb ~ L(delta) ~ 1/(delta^2 + gamma_ab^2/4). Ratio at
    # delta=0 to delta=gamma_ab/2 is exactly 2.
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    rabi = 1e-3 * emitter.gamma_ba
    g = emitter.gamma_coherence
    _, rho_bb_0, _ = emitter.steady_state_populations(omega_drive=3.0, rabi=rabi)
    _, rho_bb_half, _ = emitter.steady_state_populations(
        omega_drive=3.0 - g / 2.0, rabi=rabi
    )
    assert np.isclose(rho_bb_0 / rho_bb_half, 2.0, rtol=1e-6)


def test_populations_sum_to_one_three_level() -> None:
    emitter = ThreeLevelEmitter(
        omega_ab=3.14,
        gamma_ba=0.08,
        gamma_bc=0.03,
        gamma_ca=0.05,
        gamma_phi=0.02,
    )
    rabi = 0.2  # gives s ~ 1
    rho_aa, rho_bb, rho_cc = emitter.steady_state_populations(
        omega_drive=3.10, rabi=rabi
    )
    assert np.isclose(rho_aa + rho_bb + rho_cc, 1.0, rtol=1e-12)
    assert rho_cc > 0.0


def test_dephasing_broadens_coherence() -> None:
    emitter = ThreeLevelEmitter(
        omega_ab=3.0, gamma_ba=0.1, gamma_bc=0.05, gamma_ca=0.2, gamma_phi=0.03
    )
    assert np.isclose(
        emitter.gamma_coherence,
        emitter.gamma_ba + emitter.gamma_bc + 2.0 * emitter.gamma_phi,
        rtol=1e-12,
    )


def test_trap_configuration_raises() -> None:
    with pytest.raises(ValueError, match="trap"):
        ThreeLevelEmitter(omega_ab=3.0, gamma_ba=0.1, gamma_bc=0.05, gamma_ca=0.0)
