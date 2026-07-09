"""Optical-Bloch steady-state figure (spec Step 7).

Steady-state excited-population rho_bb vs normalized detuning delta/gamma_ab for
three drive strengths (s ~ 0.1, 1, 10): dynamiqs points on the analytic curves,
showing power broadening. Run:

    uv run python -m qubit_playground.light_matter.plot_bloch

Writes figures/obe_steady_state.png and prints the max |sim - analytic|.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qubit_playground.light_matter.bloch import steady_state
from qubit_playground.light_matter.emitter import ThreeLevelEmitter

_COLORS = ("#1f77b4", "#2ca02c", "#d62728")


def plot_steady_state(
    delta_fine: np.ndarray,
    analytic_curves: np.ndarray,
    delta_coarse: np.ndarray,
    sim_curves: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """Plot analytic rho_bb curves with dynamiqs points overlaid.

    Args:
        delta_fine: Normalized detuning delta/gamma_ab for the analytic curves.
        analytic_curves: Analytic rho_bb, shape ``(n_curves, len(delta_fine))``.
        delta_coarse: Normalized detuning for the dynamiqs points.
        sim_curves: Simulated rho_bb, shape ``(n_curves, len(delta_coarse))``.
        labels: One legend label per curve.
        output_path: Where to write the PNG.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, label in enumerate(labels):
        color = _COLORS[i % len(_COLORS)]
        ax.plot(delta_fine, analytic_curves[i], color=color, linewidth=2.0, label=label)
        ax.plot(
            delta_coarse,
            sim_curves[i],
            "o",
            color=color,
            markersize=5,
            markerfacecolor="none",
        )
    ax.set_xlabel(r"detuning  $\delta / \gamma_{ab}$")
    ax.set_ylabel(r"steady-state population  $\rho_{bb}$")
    ax.set_title("Optical Bloch steady state (lines: analytic, points: dynamiqs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _compute() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float
]:
    """Run the analytic and dynamiqs sweeps for three drive strengths."""
    emitter = ThreeLevelEmitter.two_level(omega_ab=3.0, gamma_ba=0.1)
    gamma_ab = emitter.gamma_coherence
    s_targets = [0.1, 1.0, 10.0]

    delta_fine = np.linspace(-6.0, 6.0, 201)
    delta_coarse = np.linspace(-6.0, 6.0, 13)
    analytic = np.zeros((len(s_targets), len(delta_fine)))
    sim = np.zeros((len(s_targets), len(delta_coarse)))
    labels = []
    max_err = 0.0
    for i, s in enumerate(s_targets):
        rabi = np.sqrt(s / 2.0) * emitter.gamma_ba  # s = 2*Omega0^2/Gamma_ba^2
        labels.append(rf"$s \approx {s:g}$")
        for j, d in enumerate(delta_fine):
            omega_drive = emitter.omega_ab - d * gamma_ab
            analytic[i, j] = emitter.steady_state_populations(omega_drive, rabi)[1]
        for j, d in enumerate(delta_coarse):
            omega_drive = emitter.omega_ab - d * gamma_ab
            # Off-resonance at strong drive settles at the coherence rate
            # (gamma_ab/2), slower than the population rate, so allow more
            # lifetimes than the default for a clean convergence.
            res = steady_state(
                emitter, omega_drive=omega_drive, rabi=rabi, n_lifetimes=60.0
            )
            sim[i, j] = float(res.populations[1, -1])
            ana = emitter.steady_state_populations(omega_drive, rabi)[1]
            max_err = max(max_err, abs(sim[i, j] - ana))
    return delta_fine, analytic, delta_coarse, sim, labels, max_err


def main() -> None:
    """Compute the sweeps, write the figure, and report the max error."""
    delta_fine, analytic, delta_coarse, sim, labels, max_err = _compute()
    output_path = (
        Path(__file__).resolve().parents[3] / "figures" / "obe_steady_state.png"
    )
    plot_steady_state(delta_fine, analytic, delta_coarse, sim, labels, output_path)
    print(f"Figure written to: {output_path}")
    print(f"Max |dynamiqs - analytic| rho_bb: {max_err:.3e}")


if __name__ == "__main__":
    main()
