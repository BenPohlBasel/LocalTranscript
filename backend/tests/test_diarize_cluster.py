"""Clusterung mit zu wenig Material darf nie abbrechen (Befund 2026-09-12:
ein 8-s-Clip ergab ein Fenster, AgglomerativeClustering verlangt zwei)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
from localtranscript.diarize import _cluster


def test_ein_fenster_ergibt_einen_sprecher():
    e = np.random.default_rng(0).normal(size=(1, 192))
    assert _cluster(e, None, 0.5).tolist() == [0]
    assert _cluster(e, 2, 0.5).tolist() == [0]      # gewünschte 2 Sprecher, aber nur eine Probe


def test_zwei_fenster_laufen_durch():
    e = np.random.default_rng(0).normal(size=(2, 192))
    labels = _cluster(e, None, 0.5)
    assert len(labels) == 2 and set(labels.tolist()) <= {0, 1}
