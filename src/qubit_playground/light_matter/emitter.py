"""Three-level driven emitter and its analytic steady state (paper Sec. 2).

Restates Part A.2-A.3 of the code-spec. States |a> (ground), |b> (excited),
|c> (auxiliary) with basis indices 0, 1, 2. Population decay rates Gamma_ba,
Gamma_bc out of |b> and Gamma_ca out of |c>; pure dephasing gamma_phi on the
b<->a coherence. The coherence damping rate is

    gamma_ab = Gamma_ba + Gamma_bc + 2*gamma_phi          (paper Sec. 2)

A classical field E(t) = E0 cos(omega t) drives a<->b with full Rabi amplitude
Omega0 = -d_ab*E0/hbar. In the frame rotating at the drive frequency omega the
RWA Hamiltonian is

    H_RWA = -Delta |b><b| + (Omega0/2)(|b><a| + |a><b|),   Delta = omega - omega_ab.

This module holds only the pure-NumPy analytic *reference* (paper Eqs. (9)-(10)
in the monochromatic limit); the dynamiqs simulation in bloch.py is what gets
compared against it. No JAX here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreeLevelEmitter:
    """A driven three-level emitter (paper Sec. 2, Fig. 1).

    Attributes:
        omega_ab: Transition frequency omega_ab (1/fs).
        gamma_ba: |b>->|a> population decay rate (1/fs).
        gamma_bc: |b>->|c> population decay rate (1/fs). 0 => two-level atom.
        gamma_ca: |c>->|a> population decay rate (1/fs).
        gamma_phi: Pure dephasing rate on the rho_ba coherence (1/fs).
    """

    omega_ab: float
    gamma_ba: float
    gamma_bc: float = 0.0
    gamma_ca: float = 0.0
    gamma_phi: float = 0.0

    def __post_init__(self) -> None:
        """Validate rates: non-negative, and |c> must be able to relax."""
        for name in ("gamma_ba", "gamma_bc", "gamma_ca", "gamma_phi"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.gamma_bc > 0.0 and self.gamma_ca <= 0.0:
            # Otherwise |c> is a population trap and no steady state exists
            # for the Part A.3 formulas.
            raise ValueError("gamma_bc > 0 requires gamma_ca > 0 (else |c> is a trap)")

    @property
    def gamma_coherence(self) -> float:
        """Coherence damping rate gamma_ab = Gamma_ba + Gamma_bc + 2*gamma_phi."""
        return self.gamma_ba + self.gamma_bc + 2.0 * self.gamma_phi

    @property
    def is_two_level(self) -> bool:
        """True when there is no decay into the auxiliary level |c>."""
        return self.gamma_bc == 0.0

    @classmethod
    def two_level(
        cls, omega_ab: float, gamma_ba: float, gamma_phi: float = 0.0
    ) -> ThreeLevelEmitter:
        """Construct a two-level atom (Gamma_bc = Gamma_ca = 0).

        Args:
            omega_ab: Transition frequency (1/fs).
            gamma_ba: |b>->|a> population decay rate (1/fs).
            gamma_phi: Pure dephasing rate (1/fs).

        Returns:
            A ThreeLevelEmitter with no auxiliary-level decay.
        """
        return cls(omega_ab=omega_ab, gamma_ba=gamma_ba, gamma_phi=gamma_phi)

    def _detuning(self, omega_drive: float) -> float:
        """Return delta = omega_ab - omega_drive (paper: delta = -Delta)."""
        return self.omega_ab - omega_drive

    def _lorentzian(self, delta: float) -> float:
        """L(delta) = (gamma_ab/2) / (delta^2 + gamma_ab^2/4) (Part A.3)."""
        g = self.gamma_coherence
        return (g / 2.0) / (delta**2 + g**2 / 4.0)

    def saturation(self, omega_drive: float, rabi: float) -> float:
        """Saturation parameter s (paper Eqs. (9)-(10), monochromatic limit).

        s = [2(2*Gamma_ca + Gamma_bc) / (Gamma_ca (Gamma_ba + Gamma_bc))]
            * (Omega0^2/4) * L(delta)

        In the two-level limit (Gamma_bc = 0) the Gamma_ca factors cancel and
        the prefactor reduces to 4/Gamma_ba, so this is evaluated directly to
        avoid a 0/0 when Gamma_ca = 0.

        Args:
            omega_drive: Drive angular frequency (1/fs).
            rabi: Full Rabi amplitude Omega0 (1/fs).

        Returns:
            The dimensionless saturation parameter s.
        """
        delta = self._detuning(omega_drive)
        if self.gamma_ca == 0.0:
            prefactor = 4.0 / self.gamma_ba  # two-level limit (gamma_bc == 0)
        else:
            prefactor = (
                2.0
                * (2.0 * self.gamma_ca + self.gamma_bc)
                / (self.gamma_ca * (self.gamma_ba + self.gamma_bc))
            )
        return prefactor * (rabi**2 / 4.0) * self._lorentzian(delta)

    def steady_state_coherence(self, omega_drive: float, rabi: float) -> complex:
        """Rotating-frame steady-state coherence rho_ba (Part A.3).

        rho_ba = -(Omega0/2) / (delta - 1j*gamma_ab/2) * 1/(1+s)

        This is the slowly varying (rotating-frame) coherence, directly
        comparable to <sigma_-> from the dynamiqs RWA simulation.

        Args:
            omega_drive: Drive angular frequency (1/fs).
            rabi: Full Rabi amplitude Omega0 (1/fs).

        Returns:
            The complex steady-state coherence rho_ba.
        """
        delta = self._detuning(omega_drive)
        s = self.saturation(omega_drive, rabi)
        g = self.gamma_coherence
        return -(rabi / 2.0) / (delta - 1j * g / 2.0) / (1.0 + s)

    def steady_state_populations(
        self, omega_drive: float, rabi: float
    ) -> tuple[float, float, float]:
        """Steady-state populations (rho_aa, rho_bb, rho_cc) (Part A.3).

        rho_bb = (Omega0^2/4) * gamma_ab
                 / [(Gamma_ba+Gamma_bc)(delta^2 + gamma_ab^2/4)(1+s)]
        rho_cc = (Gamma_bc/Gamma_ca) * rho_bb
        rho_aa = 1 - rho_bb - rho_cc

        Args:
            omega_drive: Drive angular frequency (1/fs).
            rabi: Full Rabi amplitude Omega0 (1/fs).

        Returns:
            The tuple (rho_aa, rho_bb, rho_cc).
        """
        delta = self._detuning(omega_drive)
        s = self.saturation(omega_drive, rabi)
        g = self.gamma_coherence
        rho_bb = (
            (rabi**2 / 4.0)
            * g
            / ((self.gamma_ba + self.gamma_bc) * (delta**2 + g**2 / 4.0) * (1.0 + s))
        )
        rho_cc = (
            (self.gamma_bc / self.gamma_ca) * rho_bb if self.gamma_bc > 0.0 else 0.0
        )
        rho_aa = 1.0 - rho_bb - rho_cc
        return rho_aa, rho_bb, rho_cc
