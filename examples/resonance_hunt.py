"""Throwaway resonance hunt for Step 6a (NOT a library dependency).

Sweeps ``relative_ldos`` vs wavelength over 500-750 nm at 1 nm steps for the
circle (rad=200 nm, n_core=3.5, pol=2, emitter 60 nm off the surface), locates
isolated whispering-gallery peaks, characterises each (omega_m, gamma_m, Q,
neighbour separation in linewidths), and cross-checks the sharpest against the
analytic Mie |b_n|^2 coefficients. Run:

    uv run python examples/resonance_hunt.py
"""

from __future__ import annotations

import numpy as np
from pysie2d import BIESolver, Geometry, Material, relative_ldos
from pysie2d.reference import mie

from qubit_playground.light_matter.units import wavelength_to_omega

RAD = 200.0
N_CORE = 3.5
N_PTS = 256
Z_S = 260.0


def main() -> None:
    """Sweep, find peaks, and print the resonance-hunt table."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    solver = BIESolver(geom, Material(n_core=N_CORE, n_clad=1.0, pol=2))

    lam = np.arange(500.0, 750.0 + 1e-9, 1.0)
    ldos = np.array([relative_ldos(solver, float(x), 0.0, Z_S) for x in lam])

    # Local maxima above a modest threshold.
    peaks = [
        i
        for i in range(1, len(lam) - 1)
        if ldos[i] > ldos[i - 1] and ldos[i] > ldos[i + 1] and ldos[i] > 1.5
    ]
    print(f"{'lam_nm':>8} {'ldos':>8} {'omega_m':>9} {'gamma_m':>10} {'Q':>8}")
    for i in peaks:
        lam_peak = lam[i]
        omega_m = float(wavelength_to_omega(lam_peak))
        # FWHM in wavelength -> gamma_m via |d omega/d lam| = omega/lam.
        half = 0.5 * (ldos[i] - 1.0) + 1.0
        lo = i
        while lo > 0 and ldos[lo] > half:
            lo -= 1
        hi = i
        while hi < len(lam) - 1 and ldos[hi] > half:
            hi += 1
        fwhm_lam = lam[hi] - lam[lo]
        gamma_m = omega_m * fwhm_lam / lam_peak
        q = omega_m / gamma_m if gamma_m > 0 else float("inf")
        print(f"{lam_peak:8.1f} {ldos[i]:8.3f} {omega_m:9.4f} {gamma_m:10.5f} {q:8.1f}")

    # Mie cross-check: |b_n|^2 vs order at the sharpest peak wavelength.
    if peaks:
        sharp = max(peaks, key=lambda i: ldos[i])
        lam_peak = float(lam[sharp])
        k = 2.0 * np.pi / lam_peak
        x = k * RAD
        m_rel = N_CORE / 1.0
        n_max = 40
        b = np.array([mie.bn(n, x, m_rel) for n in range(n_max + 1)])
        order = int(np.argmax(np.abs(b) ** 2))
        print(
            f"\nsharpest peak at {lam_peak:.1f} nm: Mie |b_n|^2 maximal at "
            f"n={order} (size parameter x={x:.3f})"
        )


if __name__ == "__main__":
    main()
