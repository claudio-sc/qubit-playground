"""Classical environment response and its emitter-dependent physics (Part A.4).

The raw spectrum S(r_s, r_s, omega) is a property of *environment + position
only*, independent of the emitter -- so ``EnvironmentSpectrum`` stores the
dimensionless S(omega) and every emitter-dependent number is a method taking the
emitter's rate(s). One classical sweep then serves unlimited quantum experiments.

The 2-D bridge normalization (Part A.4): pysie2d's ``self_green`` returns the
dimensionless S with vacuum ``Im g0(r->r) = 1/4``. Preserving vacuum consistency
gives the physical, dimensionful response

    S_tilde(omega) = 2 * Gamma_0 * S(omega)          [1/fs]

from which paper Sec. 4 carries over verbatim with S -> S_tilde:

    weak coupling (Eqs. (28)-(30)):
        gamma_p = gamma_total + 4*Gamma_0*Im S(omega_0)
        delta_omega = -2*Gamma_0*Re S(omega_0)
        consistency (Eq. (32)): gamma_p/Gamma_0 = 1 + 4*Im S = relative_ldos
    strong coupling (Eq. (34)): S(omega) ~ a/(omega_m - omega - i*gamma_m/2) + b
        g^2 = 2*Gamma_0*Re(a)  (Eq. (37)); background b renormalises the emitter.

No JAX here: this module produces plain floats/arrays for the quantum layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.fitting import fit_lorentzian
from qubit_playground.light_matter.units import wavelength_to_omega

if TYPE_CHECKING:
    from pysie2d import BIESolver


@dataclass(frozen=True)
class PhotonicMode:
    """A single photonic mode plus the emitter it dresses (Part A.4).

    Deliberately extraction-agnostic (the QNM seam): its fields are physics only
    -- mode frequency/linewidth, coupling, and dressed emitter parameters -- with
    no fit windows, residuals, or guesses (those live in ``LorentzianFit``). A
    later stage adds ``PhotonicMode.from_qnm(...)`` filling the *same* fields from
    a Beyn eigenvalue and residue integral.

    Attributes:
        omega_m: Fitted mode frequency (1/fs).
        gamma_m: Fitted mode linewidth (1/fs).
        g: Coupling constant g = sqrt(2*Gamma_0*Re a) (Eq. (37)).
        omega_emitter_dressed: omega_0' = omega_0 - 2*Gamma_0*Re b.
        gamma_emitter_dressed: gamma_0' = gamma_0 + 4*Gamma_0*Im b.
    """

    omega_m: float
    gamma_m: float
    g: float
    omega_emitter_dressed: float
    gamma_emitter_dressed: float


@dataclass(frozen=True)
class EnvironmentSpectrum:
    """Dimensionless self-Green spectrum S(r_s, r_s, omega) vs frequency.

    Attributes:
        omega: Ascending angular frequencies (1/fs).
        s_raw: Complex dimensionless S(r_s, r_s, omega) at each omega.
        x_s: Emitter x-coordinate (nm).
        z_s: Emitter z-coordinate (nm).
    """

    omega: np.ndarray
    s_raw: np.ndarray
    x_s: float
    z_s: float

    def _interp_s(self, omega_0: float) -> complex:
        """Linearly interpolate S at omega_0; raise if outside the sweep."""
        if omega_0 < self.omega[0] or omega_0 > self.omega[-1]:
            raise ValueError(
                f"omega_0={omega_0} outside sweep "
                f"[{self.omega[0]}, {self.omega[-1]}] -- refusing to extrapolate"
            )
        re = float(np.interp(omega_0, self.omega, self.s_raw.real))
        im = float(np.interp(omega_0, self.omega, self.s_raw.imag))
        return complex(re, im)

    def s_tilde(self, gamma_0: float) -> np.ndarray:
        """Dimensionful response S_tilde = 2*Gamma_0*S (Part A.4).

        Args:
            gamma_0: Free-space radiative rate Gamma_0 (1/fs).

        Returns:
            Complex S_tilde(omega), shape ``(len(omega),)``.
        """
        return 2.0 * gamma_0 * self.s_raw  # Part A.4

    def decay_rate(self, gamma_0: float, gamma_total: float, omega_0: float) -> float:
        """Environment-modified decay rate (Eq. (30)).

        gamma_p = gamma_total + 4*Gamma_0*Im S(omega_0).

        Args:
            gamma_0: Free-space radiative rate Gamma_0 (1/fs).
            gamma_total: Bare total coherence-damping rate gamma_0 (1/fs).
            omega_0: Emitter transition frequency (1/fs).

        Returns:
            The modified decay rate gamma_p (1/fs).
        """
        return gamma_total + 4.0 * gamma_0 * self._interp_s(omega_0).imag  # Eq. (30)

    def frequency_shift(self, gamma_0: float, omega_0: float) -> float:
        """Environment-induced frequency shift (Eq. (29)).

        delta_omega = -2*Gamma_0*Re S(omega_0).

        Args:
            gamma_0: Free-space radiative rate Gamma_0 (1/fs).
            omega_0: Emitter transition frequency (1/fs).

        Returns:
            The frequency shift delta_omega (1/fs).
        """
        return -2.0 * gamma_0 * self._interp_s(omega_0).real  # Eq. (29)

    def fit_mode(self, emitter: ThreeLevelEmitter) -> PhotonicMode:
        """Fit an isolated Lorentzian mode and convert it to physics (Part A.4).

        The strong-coupling mapping is derived for a two-level emitter, so this
        asserts ``emitter.is_two_level``. Uses Gamma_0 = ``emitter.gamma_ba``,
        gamma_0 = ``emitter.gamma_coherence``, omega_0 = ``emitter.omega_ab``.

        Args:
            emitter: The two-level emitter to dress.

        Returns:
            A PhotonicMode carrying mode, coupling, and dressed-emitter params.

        Raises:
            ValueError: If the emitter is not two-level, or the fit is not an
                isolated mode (Re a <= 0 or |Im a| > 0.1|Re a|).
        """
        if not emitter.is_two_level:
            raise ValueError("fit_mode requires a two-level emitter (Part A.4 mapping)")
        fit = fit_lorentzian(self.omega, self.s_raw)
        if fit.a.real <= 0.0:
            raise ValueError(f"fit is not an isolated mode: Re(a)={fit.a.real} <= 0")
        ratio = abs(fit.a.imag) / abs(fit.a.real)
        if ratio > 0.1:
            raise ValueError(
                f"fit is not an isolated mode: |Im a|/|Re a|={ratio:.3f} > 0.1"
            )

        gamma_0_rad = emitter.gamma_ba  # Gamma_0 (radiative rate)
        gamma_0 = emitter.gamma_coherence
        omega_0 = emitter.omega_ab
        g = float(np.sqrt(2.0 * gamma_0_rad * fit.a.real))  # Eq. (37)
        return PhotonicMode(
            omega_m=fit.omega_m,
            gamma_m=fit.gamma_m,
            g=g,
            omega_emitter_dressed=omega_0 - 2.0 * gamma_0_rad * fit.b.real,  # A.4
            gamma_emitter_dressed=gamma_0 + 4.0 * gamma_0_rad * fit.b.imag,  # A.4
        )


def compute_spectrum(
    solver: BIESolver,
    wavelengths_nm: np.ndarray,
    x_s: float,
    z_s: float,
) -> EnvironmentSpectrum:
    """Sweep pysie2d's self-Green function over wavelength.

    Each wavelength is a full BIE assemble+solve: unlike the LDOS *map* over
    positions (which reuses one LU factorization), there is nothing to factor
    across wavelengths, so this is an honest per-wavelength loop -- do not try
    to "optimize" it by reusing a factorization.

    Wavelengths are converted to angular frequency and the result is stored
    ascending in omega (reverse of ascending wavelength), which interpolation
    and fitting both want.

    Args:
        solver: Configured BIE solver (treated as opaque).
        wavelengths_nm: Wavelengths to sweep (nm).
        x_s: Emitter x-coordinate (nm).
        z_s: Emitter z-coordinate (nm).

    Returns:
        The EnvironmentSpectrum, ascending in omega.
    """
    from pysie2d import self_green

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    s_raw = np.array(
        [self_green(solver, float(lam), x_s, z_s) for lam in wavelengths_nm],
        dtype=complex,
    )
    omega = np.asarray(wavelength_to_omega(wavelengths_nm), dtype=float)

    order = np.argsort(omega)  # ascending omega (reverses ascending wavelength)
    return EnvironmentSpectrum(omega=omega[order], s_raw=s_raw[order], x_s=x_s, z_s=z_s)
