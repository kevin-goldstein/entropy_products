"""Fixed-theory solves, refinement checks, and finite-range matching scans."""
from pathlib import Path
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calculations"))
from reversible_branches import Settings, checked_row, fixed_coupling, residual, shoot, trace_point


def main():
    out = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1] / "results/data")) / "branches"
    out.mkdir(parents=True, exist_ok=True)
    settings = [Settings(16., 1e-11), Settings(18., 2e-12), Settings(20., 3e-14)]
    fixed = [fixed_coupling(2.99, (.2, .6), 3, setting) for setting in settings]
    assert np.ptp([r["epsilon"] for r in fixed]) < 1e-8
    assert np.ptp([r["eta"] for r in fixed]) < 1e-9
    assert abs(fixed[-1]["epsilon"] - .3468036818100) < 1e-8
    assert abs(fixed[-1]["eta"] - .9989346795981) < 1e-9
    print("[PASS] Fixed nu=2.99 recovered in three numerical settings", flush=True)

    seed = trace_point(3, .2, settings[-1])
    assert abs(seed["alpha"] - 4.893760362084) < 2e-10
    assert abs(seed["eta"] - .999859966474) < 2e-10
    # Reverse the parameterization at the recovered coupling.
    inverse = fixed_coupling(seed["nu"], (.15, .25), 3, settings[-1])
    assert abs(inverse["epsilon"] - .2) < 1e-9
    print("[PASS] Previously validated epsilon=0.2 point and fixed-coupling inverse", flush=True)

    endpoints = []
    for n in range(3, 8):
        triplet = [trace_point(n, 6., setting) for setting in settings]
        assert np.ptp([r["nu"] for r in triplet]) < 1e-8
        assert np.ptp([r["eta"] for r in triplet]) < 1e-7
        endpoints.extend(triplet)
        print(f"[PASS] n={n} endpoint refinements and full node count {n}", flush=True)

    toda = []
    for eps in [.2, 1., 3., 6.]:
        row = checked_row(shoot(2., eps, settings[-1]), 2, settings[-1])
        assert abs(row["eta"] - 1) < 2e-11
        toda.append(row)
    print("[PASS] A2 Toda controls: eta=1 over four amplitudes", flush=True)

    # Uniform sampling plus explicit sign-change bracketing is a diagnostic,
    # never a proof that roots (including tangent or closely spaced roots) are
    # absent. Both center reflection conditions are scanned separately.
    grid = np.unique(np.r_[np.geomspace(.005, .05, 9), np.linspace(.05, 6., 120)])
    probes, scan_rows = [], []
    for nu in [2.5, 2.9, 2.99, 3.01, 3.5]:
        solves = [shoot(nu, float(eps), settings[1]) for eps in grid]
        for parity in ["odd", "even"]:
            values = [residual(r, parity) for r in solves]
            roots = []
            for k in range(len(grid) - 1):
                if values[k] * values[k + 1] < 0:
                    eps = brentq(lambda e: residual(shoot(nu, e, settings[1]), parity),
                                 grid[k], grid[k + 1], xtol=5e-13)
                    found = shoot(nu, eps, settings[1])
                    n = 2 * found["half_nodes"] + int(parity == "odd")
                    roots.append(checked_row(found, n, settings[1]))
            for eps, value in zip(grid, values):
                scan_rows.append({"nu": nu, "parity": parity, "epsilon": float(eps), "matching_residual": value})
            probes.append({"nu": nu, "parity": parity, "sign_change_roots": roots,
                           "min_sample_abs_residual": min(abs(value) for value in values)})
            print(f"[PASS] nu={nu:g}, {parity}: completed {len(grid)} samples, {len(roots)} sign-change roots", flush=True)
    recovered = [r for p in probes if p["nu"] == 2.99 and p["parity"] == "odd" for r in p["sign_change_roots"]]
    assert len(recovered) == 1 and abs(recovered[0]["epsilon"] - fixed[-1]["epsilon"]) < 1e-8
    with (out / "fixed_coupling_scan.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=scan_rows[0].keys())
        writer.writeheader(); writer.writerows(scan_rows)
    report = {"settings": [setting.metadata() for setting in settings],
              "fixed_nu_2p99": fixed, "seed_epsilon_0p2": seed,
              "seed_inverse": inverse, "endpoint_refinements": endpoints,
              "A2_controls": toda, "scan_grid": grid.tolist(), "scan_results": probes,
              "scope": "Reflection-symmetric matching only. Finite scan samples epsilon in [0.005,6], excludes the trivial RN amplitude, and detects sign-changing brackets. It neither proves global uniqueness nor excludes roots between samples, tangent roots, amplitudes above six, or nonreversible solutions."}
    (out / "fixed_coupling_control.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
