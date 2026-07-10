import numpy as np

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.purcell import simulate_modified_decay


def test_extracted_pole_matches_prescription(circle_weak_spectrum) -> None:
    # The matrix-pencil pole of the coherence must equal the classical
    # prescription delta_omega - 1j*gamma_p/2. rtol=1e-3 is limited by ODE
    # accuracy and pencil conditioning over the finite window; t_final = 8/gamma_p
    # constrains both parts of the pole well.
    spectrum = circle_weak_spectrum
    omega_0 = float(spectrum.omega[len(spectrum.omega) // 2])
    emitter = ThreeLevelEmitter.two_level(omega_ab=omega_0, gamma_ba=0.01)
    gamma_p = spectrum.decay_rate(emitter.gamma_ba, emitter.gamma_coherence, omega_0)

    result = simulate_modified_decay(spectrum, emitter, t_final=8.0 / gamma_p)
    expected = result.delta_omega - 1j * result.gamma_p / 2.0
    assert np.isclose(result.extracted_pole, expected, rtol=1e-3)
    # Sign check: real part is the frequency shift, not its negative.
    assert np.sign(result.extracted_pole.real) == np.sign(result.delta_omega)


def test_enhanced_ldos_decays_faster_than_free_space(circle_weak_spectrum) -> None:
    # At the sweep frequency of largest relative LDOS (must exceed 1.3 -- widen
    # the TOML window if not), the emitter decays faster than free space. This
    # is an inequality, the physical statement the Purcell figure shows.
    spectrum = circle_weak_spectrum
    ldos = 1.0 + 4.0 * spectrum.s_raw.imag
    peak = int(np.argmax(ldos))
    assert ldos[peak] > 1.3
    omega_0 = float(spectrum.omega[peak])

    gamma_0 = 0.01  # Gamma_0
    emitter = ThreeLevelEmitter.two_level(omega_ab=omega_0, gamma_ba=gamma_0)
    t_probe = 2.0 / gamma_0
    result = simulate_modified_decay(spectrum, emitter, t_final=t_probe)

    rho_bb_sim = float(result.populations[1, -1])
    rho_bb_free = 0.5 * np.exp(-gamma_0 * t_probe)  # rho_bb(0)=0.5
    assert rho_bb_sim < rho_bb_free
