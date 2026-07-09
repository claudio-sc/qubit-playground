import numpy as np
import pytest

from qubit_playground.light_matter.scenarios import (
    EmitterPosition,
    GeometryConfig,
    MaterialConfig,
    ResonanceHint,
    ScenarioConfig,
    SweepConfig,
    compute_scenario_spectrum,
    list_scenarios,
    load_scenario,
)


@pytest.mark.parametrize("name", list_scenarios())
def test_scenario_contract(name: str) -> None:
    config = load_scenario(name)
    spectrum = compute_scenario_spectrum(config, n_points=15)  # coarse
    ldos = 1.0 + 4.0 * spectrum.s_raw.imag  # Part A.4 identity
    assert np.all(np.isfinite(ldos))
    if config.resonance is not None:
        assert ldos.max() > config.resonance.ldos_peak_min
        assert 0 < np.argmax(ldos) < len(ldos) - 1  # peak interior, not at edge


def test_load_unknown_scenario_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenario("nope")


def test_unknown_toml_key_raises(tmp_path, monkeypatch) -> None:
    # Directly exercise the strict-parsing path via the loader's helper by
    # crafting a config: an unknown key in a sub-table must be rejected.
    from qubit_playground.light_matter import scenarios

    bad = (
        "[scenario]\nname='x'\ndescription='d'\ncreated='2026'\n"
        "pysie2d_version='0.2.1'\nprovenance='p'\n"
        "[geometry]\nrad=150.0\nn_pts=256\nm=0\nlamda_typo=1\n"
        "[material]\nn_core=2.4\nn_clad=1.0\npol=2\n"
        "[emitter]\nx_s=0.0\nz_s=260.0\n"
        "[sweep]\nlambda_min_nm=580.0\nlambda_max_nm=640.0\nn_points=25\n"
    )
    path = tmp_path / "bad.toml"
    path.write_text(bad)

    class _Entry:
        name = "bad.toml"

        def read_text(self, encoding: str = "utf-8") -> str:
            return path.read_text()

    monkeypatch.setattr(scenarios, "_scenario_files", lambda: {"bad": _Entry()})
    with pytest.raises(ValueError, match="unknown key 'lamda_typo'"):
        load_scenario("bad")


def _valid_kwargs() -> dict:
    return {
        "name": "t",
        "description": "d",
        "created": "2026",
        "pysie2d_version": "0.2.1",
        "provenance": "p",
        "geometry": GeometryConfig(rad=150.0, n_pts=256, m=0),
        "material": MaterialConfig(n_core=2.4, n_clad=1.0, pol=2),
        "emitter": EmitterPosition(x_s=0.0, z_s=260.0),
        "sweep": SweepConfig(lambda_min_nm=580.0, lambda_max_nm=640.0, n_points=25),
    }


def test_validation_bad_radius() -> None:
    kw = _valid_kwargs()
    kw["geometry"] = GeometryConfig(rad=-1.0, n_pts=256, m=0)
    with pytest.raises(ValueError, match="rad must be positive"):
        ScenarioConfig(**kw)


def test_validation_too_few_boundary_points() -> None:
    kw = _valid_kwargs()
    kw["geometry"] = GeometryConfig(rad=150.0, n_pts=32, m=0)
    with pytest.raises(ValueError, match="n_pts must be >= 64"):
        ScenarioConfig(**kw)


def test_validation_bad_window_order() -> None:
    kw = _valid_kwargs()
    kw["sweep"] = SweepConfig(lambda_min_nm=640.0, lambda_max_nm=580.0, n_points=25)
    with pytest.raises(ValueError, match="lambda_min_nm must be"):
        ScenarioConfig(**kw)


def test_validation_resonance_outside_window() -> None:
    kw = _valid_kwargs()
    kw["resonance"] = ResonanceHint(lambda_peak_nm=700.0, ldos_peak_min=2.0)
    with pytest.raises(ValueError, match="lambda_peak_nm must lie inside"):
        ScenarioConfig(**kw)
