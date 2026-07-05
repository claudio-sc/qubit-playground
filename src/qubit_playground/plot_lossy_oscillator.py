"""Generate the photon-number-decay figure for the lossy harmonic oscillator.

Run with ``uv run python -m qubit_playground.plot_lossy_oscillator``. Produces
``figures/lossy_oscillator_decay.png`` comparing the dynamiqs simulation against
the analytic exponential decay, and prints the maximum absolute error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from qubit_playground.lossy_oscillator import (
    DecayResult,
    max_absolute_error,
    simulate_photon_decay,
)


def plot_decay(result: DecayResult, output_path: Path) -> None:
    """Plot simulated vs. analytic photon-number decay and save to disk.

    Args:
        result: A simulation outcome produced by ``simulate_photon_decay``.
        output_path: File path where the PNG figure is written.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        result.times,
        result.photon_number,
        label="dynamiqs simulation",
        linewidth=2.5,
        color="#1f77b4",
    )
    ax.plot(
        result.times,
        result.analytic,
        label=r"analytic  $|\alpha_0|^2 e^{-\kappa t}$",
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"mean photon number  $\langle a^\dagger a \rangle$")
    ax.set_title(f"Lossy harmonic oscillator ($\\kappa = {result.kappa}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the simulation, save the figure, and report the validation error."""
    result = simulate_photon_decay()
    output_path = (
        Path(__file__).resolve().parents[2] / "figures" / ("lossy_oscillator_decay.png")
    )
    plot_decay(result, output_path)

    error = max_absolute_error(result)
    print(f"Figure written to: {output_path}")
    print(f"Max absolute error vs. analytic decay: {error:.3e}")


if __name__ == "__main__":
    main()
