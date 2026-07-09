import dataclasses

import numpy as np
import pytest
from pysie2d import relative_ldos

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import (
    EnvironmentSpectrum,
    compute_spectrum,
)
from qubit_playground.light_matter.fitting import LorentzianFit
from qubit_playground.light_matter.scenarios import build_solver
from qubit_playground.light_matter.units import omega_to_wavelength


def test_ldos_identity(circle_weak_config, circle_weak_solver, circle_weak_spectrum):
    # Non-negotiable bridge test: with gamma_0 = gamma_total = Gamma_0, the two
    # code paths (decay_rate/Gamma_0 vs pysie2d.relative_ldos) are the *same*
    # solver output, so they must agree to machine precision (rtol=1e-12). This
    # pins the normalization chain, not the physics.
    spectrum = circle_weak_spectrum
    gamma_0 = 1.0
    x_s, z_s = circle_weak_config.emitter.x_s, circle_weak_config.emitter.z_s
    for omega in spectrum.omega:
        lam = float(omega_to_wavelength(float(omega)))
        lhs = spectrum.decay_rate(gamma_0, gamma_0, float(omega)) / gamma_0
        rhs = relative_ldos(circle_weak_solver, lam, x_s, z_s)
        assert np.isclose(lhs, rhs, rtol=1e-12)


def test_far_emitter_approaches_free_space(circle_weak_config):
    # Move the emitter to 25 radii away. In 2-D S decays only algebraically, so
    # this is a sanity bound (|s_raw| < 0.02 => decay_rate within ~8% of
    # gamma_total), not a precision test.
    config = dataclasses.replace(
        circle_weak_config,
        emitter=dataclasses.replace(circle_weak_config.emitter, z_s=5000.0),
    )
    solver = build_solver(config)
    wavelengths = np.linspace(config.sweep.lambda_min_nm, config.sweep.lambda_max_nm, 8)
    spectrum = compute_spectrum(solver, wavelengths, config.emitter.x_s, 5000.0)
    assert np.all(np.abs(spectrum.s_raw) < 0.02)
    gamma_0 = 1.0
    for omega in spectrum.omega:
        gamma_p = spectrum.decay_rate(gamma_0, gamma_0, float(omega))
        assert abs(gamma_p - gamma_0) < 0.1 * gamma_0


def test_omega_outside_sweep_raises(circle_weak_spectrum):
    with pytest.raises(ValueError, match="outside sweep"):
        circle_weak_spectrum.decay_rate(
            1.0, 1.0, float(circle_weak_spectrum.omega[-1]) + 1.0
        )


def _synth_spectrum(a: complex) -> EnvironmentSpectrum:
    omega_m, gamma_m, b = 3.0, 0.02, 0.05 - 0.01j
    model = LorentzianFit(omega_m, gamma_m, a, b)
    omega = np.linspace(omega_m - 5 * gamma_m, omega_m + 5 * gamma_m, 120)
    return EnvironmentSpectrum(
        omega=omega, s_raw=model.evaluate(omega), x_s=0.0, z_s=0.0
    )


def test_fit_mode_recovers_coupling():
    a = 0.5 + 0.0j
    spectrum = _synth_spectrum(a)
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    mode = spectrum.fit_mode(emitter)
    g_expected = np.sqrt(2.0 * emitter.gamma_ba * a.real)
    assert np.isclose(mode.g, g_expected, rtol=1e-6)


def test_fit_mode_rejects_negative_residue():
    spectrum = _synth_spectrum(-0.5 + 0.0j)
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    with pytest.raises(ValueError, match="isolated mode"):
        spectrum.fit_mode(emitter)


def test_fit_mode_rejects_three_level():
    spectrum = _synth_spectrum(0.5 + 0.0j)
    emitter = ThreeLevelEmitter(omega_ab=3.0, gamma_ba=0.1, gamma_bc=0.05, gamma_ca=0.2)
    with pytest.raises(ValueError, match="two-level"):
        spectrum.fit_mode(emitter)
