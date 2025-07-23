# Allow running tests without installing the package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from achromatcfw.io.spectrum_loader import _load_defocus
from achromatcfw.zemax_utils import fringe_metrics


def test_metrics_compute():
    data = _load_defocus()
    # Use a very small range for speed
    max_w, mean_w = fringe_metrics(data[:, 1], defocus_range=5, xrange_val=10)
    assert max_w >= 0
    assert mean_w >= 0
    assert max_w >= mean_w

    # Explicit optical conditions should match defaults
    max_w2, mean_w2 = fringe_metrics(
        data[:, 1],
        defocus_range=5,
        xrange_val=10,
        F=8.0,
        gamma=1.0,
    )
    assert max_w2 == max_w
    assert mean_w2 == mean_w
