import numpy as np

from qubit_playground.light_matter.fitting import (
    LorentzianFit,
    fit_complex_exponentials,
    fit_lorentzian,
)


def test_two_mode_recovery_is_essentially_exact() -> None:
    # Two modes: one decaying oscillation, one near-DC slow mode. Noiseless data
    # through an exact algorithm, so a loose tolerance would only mask a bug.
    dt = 0.05
    n = 400
    t = np.arange(n) * dt
    varpi = np.array([3.0 - 0.5j, 0.1 - 0.02j])  # omega - i*gamma/2
    coeffs = np.array([1.0 + 0.3j, 0.7 - 0.1j])
    signal = coeffs[0] * np.exp(-1j * varpi[0] * t) + coeffs[1] * np.exp(
        -1j * varpi[1] * t
    )
    poles = fit_complex_exponentials(signal, dt, n_modes=2)
    expected = np.sort_complex(varpi)
    np.testing.assert_allclose(np.sort_complex(poles), expected, rtol=1e-8)


def test_aliasing_when_nyquist_bound_violated() -> None:
    # dt chosen so that Re(varpi)*dt >> pi: the log branch aliases and the
    # recovered real part is wrong. This documents the failure mode.
    dt = 2.0  # Re(varpi)=3 -> 3*dt = 6 rad > pi
    n = 400
    t = np.arange(n) * dt
    varpi = np.array([3.0 - 0.05j, 0.1 - 0.02j])
    signal = np.exp(-1j * varpi[0] * t) + 0.5 * np.exp(-1j * varpi[1] * t)
    poles = fit_complex_exponentials(signal, dt, n_modes=2)
    # The fast mode's real part cannot be recovered correctly.
    assert not np.any(np.isclose(poles.real, 3.0, atol=0.1))


def test_lorentzian_round_trip() -> None:
    omega_m = 3.0
    gamma_m = 0.02
    a = 0.5 + 0.0j
    b = 0.1 - 0.05j
    truth = LorentzianFit(omega_m, gamma_m, a, b)
    omega = np.linspace(omega_m - 5 * gamma_m, omega_m + 5 * gamma_m, 200)
    s_values = truth.evaluate(omega)
    fit = fit_lorentzian(omega, s_values)
    np.testing.assert_allclose(fit.omega_m, omega_m, rtol=1e-6)
    np.testing.assert_allclose(fit.gamma_m, gamma_m, rtol=1e-6)
    np.testing.assert_allclose(fit.a, a, rtol=1e-6)
    np.testing.assert_allclose(fit.b, b, rtol=1e-6)
