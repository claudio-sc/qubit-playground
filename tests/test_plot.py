from pathlib import Path

from qubit_playground.lossy_oscillator import DecayResult


def test_plot_writes_png(result: DecayResult, tmp_path: Path) -> None:
    from qubit_playground.plot_lossy_oscillator import plot_decay

    out = tmp_path / "decay.png"
    plot_decay(result, out)
    assert out.stat().st_size > 0
