"""Declarative BIE scenario configurations (Part A.5, Step 4a).

Every place that needs a classical solver gets it from a *scenario*: a small
TOML file shipped as package data under ``scenarios/``. The code recomputes
everything from the config; nothing computed is committed. A contract test runs
every shipped scenario and checks it still delivers what it advertises.

Parsing uses the standard library only (``tomllib`` + ``importlib.resources``),
so this layer adds no dependency. Parsing is strict: an unknown key raises
``ValueError`` naming the key and the file, so a typo fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pysie2d import BIESolver

    from qubit_playground.light_matter.environment import EnvironmentSpectrum

# The data directory is a subdirectory of the light_matter package; it shares
# its name with this module, so it is reached via the parent package + joinpath
# rather than by the (shadowed) dotted name "...light_matter.scenarios".
_SCENARIO_PACKAGE = "qubit_playground.light_matter"
_SCENARIO_DIR = "scenarios"
_SCALAR_FIELDS = ("name", "description", "created", "pysie2d_version", "provenance")


@dataclass(frozen=True)
class GeometryConfig:
    """Gielis geometry parameters (passed to ``Geometry.gielis``)."""

    rad: float
    n_pts: int
    m: int


@dataclass(frozen=True)
class MaterialConfig:
    """Scatterer optical properties (passed to ``Material``)."""

    n_core: float
    n_clad: float
    pol: int


@dataclass(frozen=True)
class EmitterPosition:
    """Line-dipole / quantum-emitter position (nm)."""

    x_s: float
    z_s: float


@dataclass(frozen=True)
class SweepConfig:
    """Wavelength window for ``compute_spectrum`` (nm)."""

    lambda_min_nm: float
    lambda_max_nm: float
    n_points: int


@dataclass(frozen=True)
class ResonanceHint:
    """Advertised resonance for a resonant scenario."""

    lambda_peak_nm: float
    ldos_peak_min: float


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, validated BIE scenario.

    Attributes:
        name: Scenario name (matches the TOML file stem).
        description: One-line human description.
        created: Authoring date (ISO string).
        pysie2d_version: pysie2d version current when authored.
        provenance: How the scenario was chosen / validated.
        geometry: Gielis geometry parameters.
        material: Scatterer optical properties.
        emitter: Emitter position.
        sweep: Wavelength window.
        resonance: Optional advertised resonance (None if no [resonance] table).
    """

    name: str
    description: str
    created: str
    pysie2d_version: str
    provenance: str
    geometry: GeometryConfig
    material: MaterialConfig
    emitter: EmitterPosition
    sweep: SweepConfig
    resonance: ResonanceHint | None = None

    def __post_init__(self) -> None:
        """Validate geometry, sweep, and (if present) resonance windows."""
        if self.geometry.rad <= 0.0:
            raise ValueError("geometry.rad must be positive")
        if self.geometry.n_pts < 64:
            raise ValueError("geometry.n_pts must be >= 64")
        if self.sweep.lambda_min_nm >= self.sweep.lambda_max_nm:
            raise ValueError("sweep.lambda_min_nm must be < sweep.lambda_max_nm")
        if self.sweep.n_points < 8:
            raise ValueError("sweep.n_points must be >= 8")
        if self.resonance is not None:
            lam = self.resonance.lambda_peak_nm
            if not (self.sweep.lambda_min_nm <= lam <= self.sweep.lambda_max_nm):
                raise ValueError(
                    "resonance.lambda_peak_nm must lie inside the sweep window"
                )


def _build_strict(cls: type, table: dict[str, Any], source: str):
    """Build a dataclass from a TOML table, rejecting unknown keys.

    Args:
        cls: The dataclass to instantiate.
        table: The parsed TOML table.
        source: Name of the source file (for error messages).

    Returns:
        An instance of ``cls``.

    Raises:
        ValueError: If ``table`` contains a key not present on ``cls``.
    """
    valid = {f.name for f in fields(cls)}
    for key in table:
        if key not in valid:
            raise ValueError(
                f"unknown key '{key}' in table for {cls.__name__} ({source})"
            )
    return cls(**table)


def _scenario_files() -> dict[str, Any]:
    """Return a mapping of scenario name -> traversable TOML resource."""
    out: dict[str, Any] = {}
    scenario_dir = resources.files(_SCENARIO_PACKAGE).joinpath(_SCENARIO_DIR)
    for entry in scenario_dir.iterdir():
        if entry.name.endswith(".toml"):
            out[entry.name[: -len(".toml")]] = entry
    return out


def list_scenarios() -> list[str]:
    """Return the names of every shipped scenario TOML file (sorted).

    Returns:
        Sorted scenario names (file stems).
    """
    return sorted(_scenario_files())


def load_scenario(name: str) -> ScenarioConfig:
    """Load and validate a scenario by name.

    Args:
        name: Scenario name (TOML file stem, e.g. ``"circle_weak"``).

    Returns:
        The parsed, validated ScenarioConfig.

    Raises:
        ValueError: If ``name`` is unknown or the file has an unknown key.
    """
    files = _scenario_files()
    if name not in files:
        raise ValueError(f"unknown scenario '{name}'; valid names: {sorted(files)}")
    source = f"{name}.toml"
    raw = tomllib.loads(files[name].read_text(encoding="utf-8"))

    scenario_table = raw.get("scenario", {})
    for key in scenario_table:
        if key not in _SCALAR_FIELDS:
            raise ValueError(f"unknown key '{key}' in table [scenario] ({source})")

    resonance = None
    if "resonance" in raw:
        resonance = _build_strict(ResonanceHint, raw["resonance"], source)

    return ScenarioConfig(
        **{k: scenario_table[k] for k in _SCALAR_FIELDS},
        geometry=_build_strict(GeometryConfig, raw["geometry"], source),
        material=_build_strict(MaterialConfig, raw["material"], source),
        emitter=_build_strict(EmitterPosition, raw["emitter"], source),
        sweep=_build_strict(SweepConfig, raw["sweep"], source),
        resonance=resonance,
    )


def build_solver(config: ScenarioConfig) -> BIESolver:
    """Build a configured BIE solver from a scenario.

    Args:
        config: A validated scenario configuration.

    Returns:
        A ``pysie2d.BIESolver`` for the scenario's geometry and material.
    """
    from pysie2d import BIESolver, Geometry, Material

    geom = Geometry.gielis(
        rad=config.geometry.rad, n_pts=config.geometry.n_pts, m=config.geometry.m
    )
    material = Material(
        n_core=config.material.n_core,
        n_clad=config.material.n_clad,
        pol=config.material.pol,
    )
    return BIESolver(geom, material)


def compute_scenario_spectrum(
    config: ScenarioConfig, n_points: int | None = None
) -> EnvironmentSpectrum:
    """Build the solver and sweep the self-Green spectrum for a scenario.

    Args:
        config: A validated scenario configuration.
        n_points: Overrides ``sweep.n_points`` (e.g. for coarse test sweeps).

    Returns:
        The computed EnvironmentSpectrum.
    """
    import numpy as np

    from qubit_playground.light_matter.environment import compute_spectrum

    solver = build_solver(config)
    n = config.sweep.n_points if n_points is None else n_points
    wavelengths = np.linspace(config.sweep.lambda_min_nm, config.sweep.lambda_max_nm, n)
    return compute_spectrum(solver, wavelengths, config.emitter.x_s, config.emitter.z_s)
