"""Spectral estimators used by the classical->quantum bridge (Steps 4-6).

Two pure-NumPy/SciPy estimators, no JAX:

- ``fit_complex_exponentials``: matrix-pencil recovery of the complex
  frequencies of a sum of decaying exponentials (used to read poles out of
  dynamiqs time traces).
- ``fit_lorentzian``: least-squares fit of a single complex Lorentzian plus a
  complex background to the raw self-Green spectrum S(omega) (Part A.4,
  Eq. (34)).

Both estimators feed the quantum layer with plain floats/arrays; nothing here
touches JAX (the one-way data-flow rule of the spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def fit_complex_exponentials(signal: np.ndarray, dt: float, n_modes: int) -> np.ndarray:
    """Estimate the complex frequencies of a sum of decaying exponentials.

    Models uniform samples of ``s(t) = sum_p c_p * exp(-1j*varpi_p*t)`` and
    recovers the complex poles ``varpi_p`` by the matrix-pencil method
    (Sarkar-Pereira). The method is exact on noiseless data.

    Validity conditions (violate either and the result is silently wrong):

    - Sampling: ``dt < pi / max|Re varpi_p|``. Otherwise the ``exp(-1j*w*dt)``
      phase aliases past +-pi and the complex logarithm returns a wrong branch.
    - Length: ``len(signal) >~ 10*n_modes`` so the truncated SVD cleanly
      separates the signal subspace.

    Args:
        signal: Complex (or real) uniform samples ``s(k*dt)``, k = 0..n-1.
        dt: Sample spacing (fs).
        n_modes: Number of exponential modes to extract.

    Returns:
        Complex poles ``varpi_p`` (rad/fs), shape ``(n_modes,)``, sorted by
        real part. A pole is written ``varpi = omega - 1j*gamma/2`` so that the
        signal envelope decays as ``exp(-gamma*t/2)``.
    """
    signal = np.asarray(signal, dtype=complex)
    n = len(signal)
    pencil = n // 3
    rows = n - pencil
    # Hankel window: y[i, j] = signal[i + j].
    y = np.array([signal[i : i + pencil + 1] for i in range(rows)])
    y0, y1 = y[:, :-1], y[:, 1:]

    u, sv, vh = np.linalg.svd(y0, full_matrices=False)
    u, sv, vh = u[:, :n_modes], sv[:n_modes], vh[:n_modes]  # rank truncation

    # Eigenvalues of the reduced pencil S^-1 U^H Y1 V equal the nonzero
    # eigenvalues of the pseudoinverse pencil Y0^+ Y1; both are z_p = exp(-i w dt).
    reduced = u.conj().T @ y1 @ vh.conj().T / sv[:, None]
    z = np.linalg.eigvals(reduced)

    poles = 1j * np.log(z) / dt  # z = exp(-i*varpi*dt) => varpi = i*ln(z)/dt
    return poles[np.argsort(poles.real)]


@dataclass(frozen=True)
class LorentzianFit:
    """Result of a single-Lorentzian-plus-background fit of S(omega).

    Model: ``S(omega) = a / (omega_m - omega - 1j*gamma_m/2) + b`` (Eq. (34)).

    Attributes:
        omega_m: Fitted mode frequency (1/fs).
        gamma_m: Fitted mode linewidth (1/fs).
        a: Complex residue amplitude (a ~ real > 0 for an isolated mode).
        b: Complex background (renormalises the emitter; Part A.4).
    """

    omega_m: float
    gamma_m: float
    a: complex
    b: complex

    def evaluate(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the fitted model at given frequencies.

        Args:
            omega: Angular frequencies (1/fs).

        Returns:
            Complex model values S(omega).
        """
        return self.a / (self.omega_m - omega - 1j * self.gamma_m / 2.0) + self.b


def _initial_guess(
    omega: np.ndarray, s_values: np.ndarray
) -> tuple[float, float, complex, complex]:
    """Physically motivated starting point for the Lorentzian fit (Step 1)."""
    im = s_values.imag
    peak_idx = int(np.argmax(im))
    omega_m0 = float(omega[peak_idx])

    # Full width at half maximum of the Im S peak; gamma_m = FWHM for the model.
    half = 0.5 * im[peak_idx]
    above = im >= half
    idx = np.where(above)[0]
    span = float(omega.max() - omega.min())
    if idx.size >= 2 and idx.min() > 0 and idx.max() < len(omega) - 1:
        gamma_m0 = abs(float(omega[idx.max()] - omega[idx.min()]))
    else:
        gamma_m0 = 0.1 * span  # peak clips the window; fall back to a tenth
    if gamma_m0 <= 0.0:
        gamma_m0 = 0.1 * span

    # Background: mean of S over the outer 10% at each edge of the window.
    n_edge = max(1, len(omega) // 10)
    b0 = complex(np.mean(np.concatenate([s_values[:n_edge], s_values[-n_edge:]])))

    s_peak = complex(s_values[peak_idx])
    a0 = -1j * (gamma_m0 / 2.0) * (s_peak - b0)  # model at omega = omega_m
    return omega_m0, gamma_m0, a0, b0


def fit_lorentzian(omega: np.ndarray, s_values: np.ndarray) -> LorentzianFit:
    """Fit ``a/(omega_m - omega - 1j*gamma_m/2) + b`` to complex data.

    Uses ``scipy.optimize.least_squares`` on the stacked real/imag residuals
    with the six real parameters (omega_m, gamma_m, Re a, Im a, Re b, Im b).
    Initial guesses come from ``_initial_guess`` (peak location, half-width, and
    edge background), which is what makes the fit reliable on real spectra.

    Args:
        omega: Ascending angular frequencies (1/fs).
        s_values: Complex self-Green values S(omega) at those frequencies.

    Returns:
        A LorentzianFit with the four physical parameters.
    """
    omega = np.asarray(omega, dtype=float)
    s_values = np.asarray(s_values, dtype=complex)
    omega_m0, gamma_m0, a0, b0 = _initial_guess(omega, s_values)

    def residuals(p: np.ndarray) -> np.ndarray:
        omega_m, gamma_m, re_a, im_a, re_b, im_b = p
        model = (re_a + 1j * im_a) / (omega_m - omega - 1j * gamma_m / 2.0) + (
            re_b + 1j * im_b
        )
        diff = model - s_values
        return np.concatenate([diff.real, diff.imag])

    p0 = np.array([omega_m0, gamma_m0, a0.real, a0.imag, b0.real, b0.imag], dtype=float)
    sol = least_squares(residuals, p0, method="lm", max_nfev=10000)
    omega_m, gamma_m, re_a, im_a, re_b, im_b = sol.x
    return LorentzianFit(
        omega_m=float(omega_m),
        gamma_m=float(abs(gamma_m)),  # width is sign-agnostic in the model
        a=complex(re_a, im_a),
        b=complex(re_b, im_b),
    )
