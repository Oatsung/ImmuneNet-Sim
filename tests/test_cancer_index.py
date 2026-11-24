"""Checks that the fast CancerIndex / counters agree with a brute-force scan.

Run from the repo root:   python tests/test_cancer_index.py
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402
from app import CancerCell, CancerIndex, TCell, TissueModel  # noqa: E402

STATE = {"model": None, "queries": 0, "ties": 0}
_fast_nearest = CancerIndex.nearest


def checked_nearest(self, pos):
    """Compare each fast nearest-cancer answer with the original linear scan."""
    fast = _fast_nearest(self, pos)
    cancers = [a for a in STATE["model"].schedule.agents if isinstance(a, CancerCell)]
    if cancers:
        dist = lambda c: (pos[0] - c.pos[0]) ** 2 + (pos[1] - c.pos[1]) ** 2
        slow = min(cancers, key=dist)
        assert fast is slow, f"nearest mismatch at step {STATE['model'].schedule.steps}: {fast.unique_id} vs {slow.unique_id}"
        STATE["ties"] += sum(1 for c in cancers if dist(c) == dist(slow)) > 1
    else:
        assert fast is None
    STATE["queries"] += 1
    return fast


def check_counts(model):
    agents = list(model.schedule.agents)
    red = sum(isinstance(a, CancerCell) and not a.is_mutated for a in agents)
    orange = sum(isinstance(a, CancerCell) and a.is_mutated for a in agents)
    tcell = sum(isinstance(a, TCell) for a in agents)
    assert (model.n_normal_cancer, model.n_mutated_cancer, model.n_tcells) == (red, orange, tcell), \
        f"counter drift at step {model.schedule.steps}"
    # every live cancer cell is in the index at its true position
    idx = model.cancer_index
    assert idx.n == red + orange
    for a in agents:
        if isinstance(a, CancerCell):
            i = idx.slot[a.unique_id]
            assert (idx.x[i], idx.y[i]) == tuple(a.pos), "stale position in index"


def main():
    os.chdir(tempfile.mkdtemp())
    CancerIndex.nearest = checked_nearest
    runs = 0
    for drug in ("None", "Combination Therapy", "AI Director Adaptive"):
        for seed in range(6):
            for cfg in (dict(num_cancer=25, num_tcell=8, tcell_energy=60, base_proliferation=0.06),
                        dict(num_cancer=100, num_tcell=50, tcell_energy=200, base_proliferation=0.20)):
                with contextlib.redirect_stdout(io.StringIO()):
                    m = TissueModel(**cfg, drug_type=drug, seed=seed, autosave_csv=False)
                STATE["model"] = m
                for _ in range(60):
                    with contextlib.redirect_stdout(io.StringIO()):
                        m.step()
                    check_counts(m)
                runs += 1
    print(f"OK: {runs} runs, {STATE['queries']} nearest-cancer queries "
          f"({STATE['ties']} with distance ties) all match the brute-force scan")


if __name__ == "__main__":
    main()
