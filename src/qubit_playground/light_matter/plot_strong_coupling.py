"""Strong-coupling / vacuum-Rabi figure (spec Step 7).

Left panel: the Im S(omega) sweep with its Lorentzian fit. Right panel: the
paper-Fig-3-style anticrossing -- classical Re(varpi_pm) (lines, with a shaded
+-Im band) vs the free-space rate Gamma_0 over ~3 decades, and the quantum
matrix-pencil extractions as markers. One classical sweep feeds every point
(Part A.5). Run:

    uv run python -m qubit_playground.light_matter.plot_strong_coupling

Writes figures/vacuum_rabi.png and prints the splitting and the max
|quantum - classical| over the sweep. Setup comes from ``circle_wgm``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import EnvironmentSpectrum
from qubit_playground.light_matter.fitting import fit_lorentzian
from qubit_playground.light_matter.scenarios import (
    compute_scenario_spectrum,
    load_scenario,
)
from qubit_playground.light_matter.strong_coupling import (
    classical_eigenfrequencies,
    simulate_jaynes_cummings,
)


def plot_vacuum_rabi(
    omega: np.ndarray,
    im_s: np.ndarray,
    omega_fit: np.ndarray,
    im_s_fit: np.ndarray,
    gamma0_line: np.ndarray,
    re_classical: np.ndarray,
    im_classical: np.ndarray,
    gamma0_marker: np.ndarray,
    re_quantum: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the Lorentzian fit and the anticrossing.

    Args:
        omega: Swept angular frequencies (1/fs).
        im_s: Im S at the swept frequencies.
        omega_fit: Dense frequencies for the fitted curve.
        im_s_fit: Im of the fitted model on ``omega_fit``.
        gamma0_line: Gamma_0 values for the classical curves.
        re_classical: Classical Re(varpi_pm), shape ``(2, len(gamma0_line))``.
        im_classical: Classical Im(varpi_pm), shape ``(2, len(gamma0_line))``.
        gamma0_marker: Gamma_0 values at which the quantum sim was run.
        re_quantum: Quantum Re poles, shape ``(2, len(gamma0_marker))``.
        output_path: Where to write the PNG.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    ax0.plot(omega, im_s, "o", color="#1f77b4", markersize=4, label="pysie2d sweep")
    ax0.plot(
        omega_fit, im_s_fit, color="#d62728", linewidth=2.0, label="Lorentzian fit"
    )
    ax0.set_xlabel(r"$\omega$ (rad/fs)")
    ax0.set_ylabel(r"$\mathrm{Im}\,S(\omega)$")
    ax0.set_title("Environment spectrum and fit")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    colors = ("#1f77b4", "#d62728")
    for b in range(2):
        ax1.fill_between(
            gamma0_line,
            re_classical[b] - np.abs(im_classical[b]),
            re_classical[b] + np.abs(im_classical[b]),
            color=colors[b],
            alpha=0.15,
        )
        ax1.plot(gamma0_line, re_classical[b], color=colors[b], linewidth=2.0)
        ax1.plot(
            gamma0_marker,
            re_quantum[b],
            "o",
            color=colors[b],
            markersize=6,
            markerfacecolor="none",
        )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\Gamma_0$ (rad/fs)")
    ax1.set_ylabel(r"$\mathrm{Re}\,\varpi_\pm$ (rad/fs)")
    ax1.set_title("Vacuum Rabi anticrossing (lines: classical, points: quantum)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _mode_at(spectrum: EnvironmentSpectrum, omega_ab: float, gamma_ba: float):
    """Fit the mode for a two-level emitter of the given rate."""
    return spectrum.fit_mode(
        ThreeLevelEmitter.two_level(omega_ab=omega_ab, gamma_ba=gamma_ba)
    )


def main() -> None:
    """Compute the sweep, run the quantum markers, and write the figure."""
    spectrum = compute_scenario_spectrum(load_scenario("circle_wgm"))
    fit = fit_lorentzian(spectrum.omega, spectrum.s_raw)
    omega_m = fit.omega_m

    omega_fit = np.linspace(spectrum.omega.min(), spectrum.omega.max(), 400)
    im_s_fit = fit.evaluate(omega_fit).imag

    gamma0_line = np.logspace(-4, -1, 60) * fit.gamma_m
    re_classical = np.zeros((2, len(gamma0_line)))
    im_classical = np.zeros((2, len(gamma0_line)))
    for i, g0 in enumerate(gamma0_line):
        mode = _mode_at(spectrum, omega_m, float(g0))
        lower, upper = classical_eigenfrequencies(mode)
        re_classical[:, i] = [lower.real, upper.real]
        im_classical[:, i] = [lower.imag, upper.imag]

    gamma0_marker = np.logspace(-4, -1, 10) * fit.gamma_m
    re_quantum = np.zeros((2, len(gamma0_marker)))
    max_err = 0.0
    for i, g0 in enumerate(gamma0_marker):
        mode = _mode_at(spectrum, omega_m, float(g0))
        poles_c = classical_eigenfrequencies(mode)
        # Floor the decay rate: at weak coupling the emitter pole is nearly
        # undamped, so an uncapped 4/gamma_amp window would undersample the fixed
        # gamma_m dynamics. A gamma_m-scaled floor keeps dt well inside Nyquist.
        gamma_amp = max(min(abs(p.imag) for p in poles_c), 0.05 * mode.gamma_m)
        result = simulate_jaynes_cummings(mode, t_final=4.0 / gamma_amp, n_times=2000)
        re_quantum[:, i] = [p.real for p in result.poles_quantum]
        for pq, pc in zip(result.poles_quantum, poles_c, strict=True):
            max_err = max(max_err, abs(pq - pc))

    output_path = Path(__file__).resolve().parents[3] / "figures" / "vacuum_rabi.png"
    plot_vacuum_rabi(
        spectrum.omega,
        spectrum.s_raw.imag,
        omega_fit,
        im_s_fit,
        gamma0_line,
        re_classical,
        im_classical,
        gamma0_marker,
        re_quantum,
        output_path,
    )
    splitting = abs(re_classical[1, -1] - re_classical[0, -1])
    print(f"Figure written to: {output_path}")
    print(f"Max classical Re-splitting over sweep: {splitting:.4e} rad/fs")
    print(f"Max |quantum - classical| pole over markers: {max_err:.3e}")


if __name__ == "__main__":
    main()
