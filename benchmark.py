"""Scaling benchmark for TissueModel.

Runs the same seeded simulation on progressively larger grids (cell counts scale
with grid area) and reports wall-clock time per run and per step.

    python benchmark.py                 # default sizes 20, 40, 80, 120
    python benchmark.py 20 60 100       # custom grid sizes
"""
import contextlib
import inspect
import io
import os
import sys
import tempfile
import time

from app import TissueModel

STEPS = 60
CANCER_DENSITY = 0.08   # cancer cells per grid site at t=0
TCELL_DENSITY = 0.04    # T-cells per grid site at t=0


def run_once(grid_size, seed=0):
    area = grid_size * grid_size
    kwargs = dict(
        num_cancer=int(area * CANCER_DENSITY),
        num_tcell=int(area * TCELL_DENSITY),
        tcell_energy=100,
        base_proliferation=0.08,
        drug_type="Combination Therapy",
        seed=seed,
        grid_size=grid_size,
    )
    # The CSV autosave is only useful for the GUI; switch it off when supported.
    if "autosave_csv" in inspect.signature(TissueModel.__init__).parameters:
        kwargs["autosave_csv"] = False

    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        model = TissueModel(**kwargs)
        for _ in range(STEPS):
            model.step()
        elapsed = time.perf_counter() - t0
    return elapsed, len(model.schedule.agents)


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [20, 40, 80, 120]
    os.chdir(tempfile.mkdtemp())  # keep any simulation_data.csv out of the repo
    print(f"{'grid':>6} {'start cells':>12} {'end agents':>11} {'time (s)':>9} {'ms/step':>9}")
    for g in sizes:
        elapsed, n_end = run_once(g)
        start = int(g * g * (CANCER_DENSITY + TCELL_DENSITY))
        print(f"{g:>4}x{g:<2} {start:>12} {n_end:>11} {elapsed:>9.2f} {1000 * elapsed / STEPS:>9.1f}")
