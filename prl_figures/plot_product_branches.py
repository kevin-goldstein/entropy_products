"""Plot the PRL branch comparison from supplied reference data.

Run: python3 plot_product_branches.py
Dependencies: numpy and matplotlib. No differential equations are solved.
The sole figure export is PDF.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

_cache = tempfile.TemporaryDirectory(prefix="hot-attractors-prl-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name)
os.environ.setdefault("XDG_CACHE_HOME", _cache.name)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"
COLORS = ["#1f5f95", "#b75d1b", "#218477", "#964b88", "#697b20"]


def main():
    symmetric_source = DATA / "branch_trace.csv"
    asymmetric_source = DATA / "branch_r2_n2.csv"
    with symmetric_source.open() as stream:
        rows = list(csv.DictReader(stream))
    asym = np.genfromtxt(asymmetric_source, delimiter=",", names=True)
    assert len(rows) == 225
    assert len(asym) == 71
    assert np.isfinite(np.c_[asym["phi_plus"], asym["nu"], asym["eta"]]).all()
    assert np.count_nonzero(asym["phi_plus"] == 0) == 1
    assert np.allclose(asym["phi_plus"][[0, -1]], [-3, 3])
    nu_example = (np.sqrt(29) - 1) / 2
    eta_example = 1.0220531880189

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": .7,
        "xtick.major.width": .7,
        "ytick.major.width": .7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    fig.subplots_adjust(left=.078, right=.99, bottom=.21, top=.88, wspace=.28)
    ax, bx = axes
    endpoints = []
    for n, color in zip(range(3, 8), COLORS):
        branch = sorted((r for r in rows if int(r["n"]) == n),
                        key=lambda r: float(r["epsilon"]))
        assert len(branch) == 45
        assert all(int(r["nodes"]) == n for r in branch)
        assert float(branch[-1]["epsilon"]) == 6
        nu = np.r_[float(n), [float(r["nu"]) for r in branch]]
        eta = np.r_[1., [float(r["eta"]) for r in branch]]
        assert np.all(eta[1:] < 1)
        ax.plot(nu, eta, color=color, lw=1.5, solid_capstyle="round")
        ax.plot(nu[0], eta[0], "o", ms=3.7, mfc="white", mec=color, mew=1.0, zorder=4)
        ax.plot(nu[-1], eta[-1], "o", ms=3.7, color=color, zorder=4)
        # Direct labels distinguish the curves without requiring color vision.
        ax.text(n, 1.006, rf"$n={n}$", color=color, ha="center", va="bottom", fontsize=8.5)
        if n == 3:
            # Mark the direction using two retained samples, not interpolated data.
            arrow_start = min(branch, key=lambda r: abs(float(r["epsilon"]) - .7))
            arrow_end = min(branch, key=lambda r: abs(float(r["epsilon"]) - 1.2))
            ax.annotate("",
                        xy=(float(arrow_end["nu"]), float(arrow_end["eta"])),
                        xytext=(float(arrow_start["nu"]), float(arrow_start["eta"])),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.4,
                                    "color": color, "mutation_scale": 10},
                        zorder=6)
            ax.text(3.43, .949, "increasing\n" + r"$\epsilon$",
                    fontsize=8, color=color, ha="center", va="center",
                    linespacing=.9)
        endpoints.append({"n": n, "phi_plus": 6., "nu": float(nu[-1]), "eta": float(eta[-1])})
    ax.set_xlim(2.74, 7.25)
    ax.set_ylim(.885, 1.026)
    ax.set_xticks(range(3, 8))
    ax.set_yticks([.90, .95, 1.])
    ax.set_title("(a) Equal slope magnitudes", loc="left", pad=8)

    bx.plot(asym["nu"], asym["eta"], color=COLORS[0], lw=1.6, zorder=3)
    bx.plot([2], [1], "o", ms=4.3, mfc="white", mec=COLORS[0], mew=1.0, zorder=5)
    bx.plot(asym["nu"][[0, -1]], asym["eta"][[0, -1]], "o", ms=3.7,
            color=COLORS[0], zorder=4)
    bx.plot([nu_example], [eta_example], "*", ms=8.5, color="#9a3e11",
            mec="white", mew=.4, zorder=6)
    bx.annotate("fixed example", xy=(nu_example, eta_example),
                xytext=(1.87, 1.125), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "lw": .6, "color": ".35"},
                ha="left", va="center")
    bx.annotate("Reissner–Nordström\nlimit " + r"($\nu=2$)", xy=(2., 1.),
                xytext=(2.14, .949), fontsize=8.1,
                arrowprops={"arrowstyle": "-", "lw": .6, "color": ".35",
                            "shrinkA": 3, "shrinkB": 4},
                ha="center", va="center", zorder=4)
    bx.text(2.35, 1.184, "excess", ha="right", va="center", fontsize=8.5)
    bx.text(1.84, .923, "deficit", ha="left", va="center", fontsize=8.5)
    bx.set_xlim(1.76, 2.40)
    bx.set_ylim(.895, 1.235)
    bx.set_xticks([1.8, 2.0, 2.2, 2.4])
    bx.set_yticks([.9, 1., 1.1, 1.2])
    bx.set_title(r"(b) Unequal slopes: $(s,-2s)$", loc="left", pad=8)

    for a in axes:
        a.axhline(1., color=".55", lw=.7, ls=(0, (3, 3)), zorder=0)
        a.set_xlabel(r"$\nu$", labelpad=2)
        a.set_ylabel(r"$\eta=S_+S_-/S_{\mathrm{ext}}^2$", labelpad=4)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=".9", lw=.5)
        a.set_axisbelow(True)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "product_branches.pdf")
    plt.close(fig)

    metadata = {
        "sources": [
            {"file": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in [symmetric_source, asymmetric_source]
        ],
        "new_numerical_solutions": False,
        "generator": {
            "file": Path(__file__).name,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "export_formats": ["pdf"],
        "annotation_changes": {
            "symmetric_direction": {
                "branch_nodes": 3, "epsilon_start": .7, "epsilon_end": 1.2,
                "label": "increasing epsilon",
            },
            "asymmetric_origin": {
                "nu": 2., "eta": 1.,
                "label": "Reissner–Nordström limit (nu=2)",
                "continuous_branch_retained": True,
            },
        },
        "canvas_inches": [7.1, 2.3],
        "left": {
            "charges": "Q1=Q2=1",
            "slopes": "(alpha,-alpha), alpha^2=2 nu(nu+1)",
            "nonzero_samples_per_branch": 45,
            "nodes": [3, 4, 5, 6, 7],
            "zero_amplitude": "Analytic limits (nu,eta)=(n,1), inserted as open markers.",
            "finite_last_samples": endpoints,
        },
        "right": {
            "charges_squared": [2, 1],
            "slopes": "(s,-2s)",
            "nodes": 2,
            "samples": 71,
            "nonzero_numerical_samples": 70,
            "zero_amplitude": "Analytic point (phi_plus,nu,eta)=(0,2,1).",
            "finite_amplitude_range": [-3, 3],
            "fixed_example": {
                "slopes": ["sqrt(7)", "-2 sqrt(7)"],
                "phi_plus": 0.4427936622730,
                "nu": float(nu_example),
                "eta": eta_example,
            },
        },
        "markers": "Open circles: analytic zero-amplitude limits; filled circles: finite last samples; star: independently checked fixed-slope excess example.",
        "interpolation": "Straight line segments between retained samples; no extrapolation or fitted continuation.",
        "endpoint_scope": "Finite sampled amplitudes are not established endpoints of complete branches.",
    }
    (DATA / "product_branches_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Generated product_branches.pdf from unchanged reference data.")


if __name__ == "__main__":
    main()
