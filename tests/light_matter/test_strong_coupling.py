import numpy as np
import pytest

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import EnvironmentSpectrum, PhotonicMode
from qubit_playground.light_matter.scenarios import (
    compute_scenario_spectrum,
    load_scenario,
)
from qubit_playground.light_matter.strong_coupling import (
    classical_eigenfrequencies,
    simulate_jaynes_cummings,
)


@pytest.fixture(scope="module")
def wgm_spectrum() -> EnvironmentSpectrum:
    # Most expensive single computation in the suite: ~60 BIE solves. Computed
    # once; each test refits (milliseconds) its own emitter.
    return compute_scenario_spectrum(load_scenario("circle_wgm"))


def _mode_at(spectrum: EnvironmentSpectrum, gamma_ba: float) -> PhotonicMode:
    om = spectrum.fit_mode(
        ThreeLevelEmitter.two_level(
            omega_ab=float(spectrum.omega.mean()), gamma_ba=1e-3
        )
    ).omega_m
    emitter = ThreeLevelEmitter.two_level(omega_ab=om, gamma_ba=gamma_ba)
    return spectrum.fit_mode(emitter)


def _strong_coupling_mode(spectrum: EnvironmentSpectrum) -> PhotonicMode:
    # Never hard-code Gamma_0: find the strong-coupling regime by doubling from
    # 1e-4*gamma_m until 2g > 3*(gamma_0' + gamma_m)/2.
    gamma_m = _mode_at(spectrum, 1e-3).gamma_m
    gamma_0 = 1e-4 * gamma_m
    for _ in range(60):
        mode = _mode_at(spectrum, gamma_0)
        if 2.0 * mode.g > 1.5 * (mode.gamma_emitter_dressed + mode.gamma_m):
            return mode
        gamma_0 *= 2.0
    raise AssertionError("no strong-coupling regime found")


def test_flagship_quantum_matches_classical(wgm_spectrum) -> None:
    # The headline: quantum (matrix-pencil) and classical (Eq. (37)) complex
    # eigenfrequencies agree for both polaritons. Both sides use the same
    # (omega_m, gamma_m, g, gamma_0', omega_0'), and the <=1-excitation dynamics
    # closes exactly on the 2x2 matrix whose eigenvalues ARE varpi_pm, so the
    # comparison is exact-in-principle. The spec allows tightening from the
    # generous 1e-2 if agreement is observed far better: the clean, noiseless,
    # exactly-two-mode dynamiqs signal lets the pencil recover the poles to ~1e-15
    # here, so 1e-3 is used -- still ~12 orders above the observed floor, robust
    # to BLAS/JIT variation, and not pinned to the observed value.
    mode = _strong_coupling_mode(wgm_spectrum)
    poles_c = classical_eigenfrequencies(mode)
    gamma_amp = min(abs(p.imag) for p in poles_c)  # slowest amplitude decay
    t_final = 4.0 / gamma_amp
    result = simulate_jaynes_cummings(mode, t_final=t_final, n_times=1500)

    for pq, pc in zip(result.poles_quantum, poles_c, strict=True):
        assert np.isclose(pq, pc, rtol=1e-3)


def test_splitting_is_observable(wgm_spectrum) -> None:
    # Observable vacuum Rabi splitting (paper Sec. 4C): the real splitting
    # exceeds the summed half-linewidths.
    mode = _strong_coupling_mode(wgm_spectrum)
    lower, upper = classical_eigenfrequencies(mode)
    real_split = abs(upper.real - lower.real)
    imag_sum = abs(upper.imag) + abs(lower.imag)
    assert real_split > imag_sum


def test_weak_limit_recovers_bare_poles(wgm_spectrum) -> None:
    # Same machinery, Gamma_0 1000x smaller: the two quantum poles collapse to
    # the bare emitter (varpi_0') and mode (varpi_m) separately -- continuity
    # with the Purcell regime of Step 5.
    strong = _strong_coupling_mode(wgm_spectrum)
    # Reconstruct the strong Gamma_0 from g^2 = 2*Gamma_0*Re(a) is unnecessary;
    # rebuild directly via the doubling result's implied scale.
    gamma_m = strong.gamma_m
    # Find the strong Gamma_0 by repeating the loop's stopping scale.
    gamma_0 = 1e-4 * gamma_m
    while 2.0 * _mode_at(wgm_spectrum, gamma_0).g <= 1.5 * (
        _mode_at(wgm_spectrum, gamma_0).gamma_emitter_dressed + gamma_m
    ):
        gamma_0 *= 2.0
    weak_mode = _mode_at(wgm_spectrum, gamma_0 / 1e3)

    varpi_0p = (
        weak_mode.omega_emitter_dressed - 1j * weak_mode.gamma_emitter_dressed / 2.0
    )
    varpi_m = weak_mode.omega_m - 1j * weak_mode.gamma_m / 2.0
    target = sorted([varpi_0p, varpi_m], key=lambda z: z.imag)

    poles_c = classical_eigenfrequencies(weak_mode)
    gamma_amp = min(abs(p.imag) for p in poles_c)
    result = simulate_jaynes_cummings(weak_mode, t_final=4.0 / gamma_amp, n_times=1500)
    got = sorted(result.poles_quantum, key=lambda z: z.imag)
    for pq, tgt in zip(got, target, strict=True):
        assert np.isclose(pq, tgt, rtol=1e-2)


def test_undersampled_dt_raises(wgm_spectrum) -> None:
    mode = _strong_coupling_mode(wgm_spectrum)
    with pytest.raises(ValueError, match="undersamples"):
        simulate_jaynes_cummings(mode, t_final=1e6, n_times=10)
