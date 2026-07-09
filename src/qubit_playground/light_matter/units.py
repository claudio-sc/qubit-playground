"""Unit system for the light-matter bridge (Part A.1 of the code-spec).

The classical solver ``pysie2d`` works in nanometres; the quantum side works in
angular frequency. This module fixes the one-and-only unit system so no other
module hard-codes the speed of light:

- Length: nm. Time: fs. Angular frequency and all rates: rad/fs (written 1/fs).
- ``C_NM_PER_FS`` is the speed of light in nm/fs.
- ``omega = 2*pi * C_NM_PER_FS / lambda_nm`` and its inverse.

A lambda = 600 nm transition maps to omega ~ 3.14 rad/fs -- O(1) numbers, which
keeps both the ODE solver and the least-squares fits well conditioned.
"""

from __future__ import annotations

import numpy as np

C_NM_PER_FS: float = 299.792458
"""Speed of light in nm/fs."""


def wavelength_to_omega(lambda_nm: float | np.ndarray) -> float | np.ndarray:
    """Convert free-space wavelength to angular frequency.

    Args:
        lambda_nm: Free-space wavelength(s) in nm.

    Returns:
        Angular frequency omega = 2*pi*c/lambda in rad/fs.
    """
    return 2.0 * np.pi * C_NM_PER_FS / lambda_nm


def omega_to_wavelength(omega: float | np.ndarray) -> float | np.ndarray:
    """Convert angular frequency to free-space wavelength.

    Args:
        omega: Angular frequency(ies) in rad/fs.

    Returns:
        Free-space wavelength lambda = 2*pi*c/omega in nm.
    """
    return 2.0 * np.pi * C_NM_PER_FS / omega
