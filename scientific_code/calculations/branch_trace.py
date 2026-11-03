"""Trace the five symmetric regular branches through finite amplitude six."""
from pathlib import Path
import csv
import json
import os

import numpy as np

from reversible_branches import Settings, trace_point


def main():
    out = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1] / "results/data")) / "branches"
    out.mkdir(parents=True, exist_ok=True)
    amplitudes = np.unique(np.r_[np.linspace(.05, .5, 10), np.linspace(.6, 2., 15), np.linspace(2.2, 6., 20)])
    settings = Settings()
    rows, trends = [], []
    for n in range(3, 8):
        branch = [trace_point(n, float(eps), settings) for eps in amplitudes]
        # Record every sample, including roundoff-scale reversals on the
        # large-amplitude plateaus. Do not fit or enforce monotonicity.
        delta_nu = np.diff([r["nu"] for r in branch])
        delta_eta = np.diff([r["eta"] for r in branch])
        trends.append({"n": n,
                       "largest_nu_increase": float(max(0., max(delta_nu))),
                       "largest_eta_increase": float(max(0., max(delta_eta))),
                       "nu_nondecreasing_steps": int(sum(delta_nu >= 0)),
                       "eta_nondecreasing_steps": int(sum(delta_eta >= 0))})
        assert all(0 < r["eta"] < 1 for r in branch)
        rows.extend(branch)
        print(f"[PASS] n={n}: {len(branch)} samples; epsilon=6 nu={branch[-1]['nu']:.12f}, eta={branch[-1]['eta']:.12f}", flush=True)
    with (out / "branch_trace.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    metadata = {"model": "V_eff=2*cosh(alpha*phi), alpha**2=2*nu*(nu+1), V_g=0",
                "settings": settings.metadata(), "modes": list(range(3, 8)),
                "amplitudes": amplitudes.tolist(), "sample_count": len(rows),
                "node_count": "zeros on full open interhorizon interval, completed by reflection",
                "center_zero_exclusion": 1e-6,
                "scope": "Finite-amplitude reversible branches only. Lower sampled couplings are not certified endpoints; deficits are not certified maxima. All raw samples and any precision-scale reversals are retained. Trends do not prove continuous global monotonicity; nonreversible solutions and other roots are not excluded.",
                "sample_trends": trends,
                "max_energy_error": max(abs(r["energy_error"]) for r in rows),
                "max_matching_residual": max(abs(r["matching_residual"]) for r in rows),
                "endpoint_samples": [r for r in rows if r["epsilon"] == 6.]}
    (out / "branch_trace.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
