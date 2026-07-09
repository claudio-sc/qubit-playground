"""Purcell-effect figure (spec Step 7).

Left panel: relative-LDOS map around the cylinder (emitter marked, NaNs masked).
Right panel: rho_bb(t) in free space vs next to the cylinder, with the
classical-rate exponentials dashed. Run:

    uv run python -m qubit_playground.light_matter.plot_purcell

Writes figures/purcell_decay.png and prints (gamma_p/Gamma_0, delta_omega/Gamma_0).
All solver setup comes from the ``circle_weak`` scenario.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import compute_spectrum
from qubit_playground.light_matter.purcell import simulate_modified_decay
from qubit_playground.light_matter.scenarios import build_solver, load_scenario
from qubit_playground.light_matter.units import omega_to_wavelength


def plot_purcell(
    ldos_map: np.ndarray,
    extent: tuple[float, float, float, float],
    boundary_x: np.ndarray,
    boundary_z: np.ndarray,
    emitter_xz: tuple[float, float],
    times: np.ndarray,
    rho_bb_near: np.ndarray,
    rho_bb_free: np.ndarray,
    classical_near: np.ndarray,
    classical_free: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the LDOS map and the free-space vs near-cylinder decay.

    Args:
        ldos_map: Relative-LDOS grid (NaN at invalid positions).
        extent: (x_min, x_max, z_min, z_max) for the map (nm).
        boundary_x: Cylinder boundary x-coordinates (nm).
        boundary_z: Cylinder boundary z-coordinates (nm).
        emitter_xz: Emitter (x, z) position (nm).
        times: Save times for the decay curves (fs).
        rho_bb_near: Simulated rho_bb next to the cylinder.
        rho_bb_free: Simulated rho_bb in free space.
        classical_near: Classical exponential 0.5*exp(-gamma_p*t).
        classical_free: Classical exponential 0.5*exp(-Gamma_0*t).
        output_path: Where to write the PNG.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    masked = np.ma.masked_invalid(ldos_map)
    im = ax0.imshow(
        masked, extent=extent, origin="lower", aspect="equal", cmap="inferno"
    )
    ax0.plot(boundary_x, boundary_z, color="white", linewidth=1.0)
    ax0.plot(*emitter_xz, "co", markersize=8, label="emitter")
    ax0.set_xlabel("x (nm)")
    ax0.set_ylabel("z (nm)")
    ax0.set_title("Relative LDOS around the cylinder")
    ax0.legend(loc="upper right")
    fig.colorbar(im, ax=ax0, label=r"$\rho / \rho_0$")

    ax1.plot(times, rho_bb_near, color="#d62728", linewidth=2.0, label="near cylinder")
    ax1.plot(times, rho_bb_free, color="#1f77b4", linewidth=2.0, label="free space")
    ax1.plot(times, classical_near, "--", color="#d62728", linewidth=1.0)
    ax1.plot(times, classical_free, "--", color="#1f77b4", linewidth=1.0)
    ax1.set_xlabel("time (fs)")
    ax1.set_ylabel(r"$\rho_{bb}(t)$")
    ax1.set_title("Modified spontaneous decay (dashed: classical rates)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Compute the LDOS map and decay curves, then write the figure."""
    from pysie2d import relative_ldos_map

    config = load_scenario("circle_weak")
    solver = build_solver(config)
    wavelengths = np.linspace(
        config.sweep.lambda_min_nm, config.sweep.lambda_max_nm, config.sweep.n_points
    )
    spectrum = compute_spectrum(
        solver, wavelengths, config.emitter.x_s, config.emitter.z_s
    )
    ldos = 1.0 + 4.0 * spectrum.s_raw.imag
    peak = int(np.argmax(ldos))
    omega_0 = float(spectrum.omega[peak])
    lam = float(omega_to_wavelength(omega_0))

    rad = config.geometry.rad
    span = 2.0 * rad
    grid = np.linspace(-span, span, 161)
    xg, zg = np.meshgrid(grid, grid)
    ldos_grid = relative_ldos_map(solver, lam, xg, zg)

    gamma_0 = 0.01
    emitter = ThreeLevelEmitter.two_level(omega_ab=omega_0, gamma_ba=gamma_0)
    t_final = 3.0 / gamma_0
    result = simulate_modified_decay(spectrum, emitter, t_final=t_final)
    times = np.asarray(result.times)
    rho_bb_near = np.asarray(result.populations[1])
    rho_bb_free = 0.5 * np.exp(-gamma_0 * times)
    classical_near = 0.5 * np.exp(-result.gamma_p * times)

    geom = solver.geometry
    boundary_x = np.append(np.asarray(geom.f), geom.f[0])
    boundary_z = np.append(np.asarray(geom.g), geom.g[0])

    output_path = Path(__file__).resolve().parents[3] / "figures" / "purcell_decay.png"
    plot_purcell(
        ldos_grid,
        (-span, span, -span, span),
        boundary_x,
        boundary_z,
        (config.emitter.x_s, config.emitter.z_s),
        times,
        rho_bb_near,
        rho_bb_free,
        classical_near,
        rho_bb_free,
        output_path,
    )
    print(f"Figure written to: {output_path}")
    print(
        f"gamma_p/Gamma_0 = {result.gamma_p / gamma_0:.3f}, "
        f"delta_omega/Gamma_0 = {result.delta_omega / gamma_0:.3f}"
    )


if __name__ == "__main__":
    main()
