#!/usr/bin/env python3
"""Reproduce three distinct Supplemental Material figures from unchanged reference data.

No differential equations are solved. The profiles, branch samples and family
metadata were copied from code snapshot 2026.11.02-r5; hashes are recorded in
figure_data/supplement_figures_metadata.json. The only scientific-data scalar
not in these plotting files is the asymptotic modulus, quoted to nine decimals
from validation/excess/asymmetric_excess.log in that snapshot.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

_cache = tempfile.TemporaryDirectory(prefix="hot-attractors-supplement-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name)
os.environ.setdefault("XDG_CACHE_HOME", _cache.name)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
provenance = json.loads((DATA / "supplement_figures_metadata.json").read_text())
for name, expected in provenance["source_hashes"].items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Reference data hash mismatch: {name}")

interior = np.loadtxt(DATA / "profile_interior.csv", delimiter=",", skiprows=1)
exterior = np.loadtxt(DATA / "profile_exterior.csv", delimiter=",", skiprows=1)
b_int, p_int = interior[:, 1], interior[:, 2]
b_ext, p_ext = exterior[:, 1], exterior[:, 2]
bm, B = b_int[0], b_int[-1]
eps = p_int[-1]
phi_inf = provenance["asymptotic_modulus"]

# ---------- scalar profile: no repeated entropy-product branch panel
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 9, "axes.labelsize": 10,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
fig, ax = plt.subplots(figsize=(3.9, 2.8))
fig.subplots_adjust(left=.15, right=.97, bottom=.20, top=.96)
bb = np.concatenate([b_int, b_ext])
pp = np.concatenate([p_int, p_ext])
ax.axhline(0., color=GRID, lw=.8, zorder=0)
ax.axhline(phi_inf, color=INK2, lw=.8, ls=(0, (4, 3)), zorder=1)
for bh in (bm, B):
    ax.axvline(bh, color=INK2, lw=.7, ls=(0, (1, 2)), zorder=1)
ax.plot(bb, pp, color=BLUE, lw=1.6, zorder=3)
ax.set_xscale("log")
ax.set_xlim(.45, 4e4)
ax.set_ylim(-.32, 1.08)
ax.set_xlabel("areal radius $b$")
ax.set_ylabel(r"scalar $\phi$")
ax.text(bm*1.08, -.29, r"$b_-$", color=INK2, va="bottom")
ax.text(B*1.08, -.29, r"$b_+$", color=INK2, va="bottom")
ax.text(3e4, phi_inf-.03, r"$\phi_\infty=%.3f$" % phi_inf,
        color=INK2, ha="right", va="top")
ax.text(30, .47, r"$\phi_-=\phi_+=%.4f$" % eps + "\n(both horizons)",
        color=INK, fontsize=8.5, va="center")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, color=GRID, lw=.5, zorder=0)
ax.set_axisbelow(True)
fig.savefig(FIG / "scalar_profile.pdf", metadata={
    "Title": "Scalar profile of the unequal slope excess example",
    "Subject": "Unchanged interior and exterior reference profile",
    "CreationDate": None, "ModDate": None,
})
plt.close(fig)

# ---------- amplitude example: the two supplemental plots absent from the Letter
from matplotlib.lines import Line2D
with (DATA / "branch_trace.csv").open() as fp:
    rows = list(csv.DictReader(fp))
branch = sorted((r for r in rows if int(r["n"]) == 3),
                key=lambda r: float(r["epsilon"]))
assert len(branch) == 45
assert all(int(r["nodes"]) == 3 for r in branch)
assert float(branch[-1]["epsilon"]) == 6.
amplitude = np.r_[0., [float(r["epsilon"]) for r in branch]]
nu = np.r_[3., [float(r["nu"]) for r in branch]]
eta = np.r_[1., [float(r["eta"]) for r in branch]]
assert np.all(eta[1:] < 1.)
color = "#1f5f95"
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65))
fig.subplots_adjust(left=.095, right=.98, bottom=.30, top=.87, wspace=.43)
for ax, y, label, title in zip(axes, [nu, eta], [r"$\nu$", r"$\eta$"],
                              ["(a) Coupling parameter", "(b) Entropy product"]):
    ax.plot(amplitude, y, color=color, lw=1.7)
    ax.plot(amplitude[0], y[0], "o", ms=4, mfc="white", mec=color, mew=1.1)
    ax.plot(amplitude[-1], y[-1], "o", ms=4, color=color)
    ax.set_xlabel(r"$\phi_+$")
    ax.set_ylabel(label)
    ax.set_xlim(-.12, 6.15)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_title(title, fontsize=9, loc="left", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.5)
    ax.set_axisbelow(True)
axes[0].set_yticks([2.95, 2.975, 3.])
axes[0].set_ylim(2.948, 3.003)
axes[1].set_yticks([.965, .98, 1.])
axes[1].set_ylim(.961, 1.002)
handles = [
    Line2D([], [], marker="o", mfc="white", mec=color, color="none", ms=4,
           label=r"Analytic limit: $\phi_+\to0$"),
    Line2D([], [], marker="o", mfc=color, mec=color, color="none", ms=4,
           label=r"Last sample: $\phi_+=6$"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .015),
           ncol=2, frameon=False, fontsize=8.5)
fig.savefig(FIG / "amplitude_example.pdf", metadata={
    "Title": "Coupling and entropy product versus horizon amplitude on the n=3 branch",
    "Subject": "Unchanged 45-sample symmetric branch with analytic zero-amplitude limit",
    "CreationDate": None, "ModDate": None,
})
plt.close(fig)

OUTPUT = FIG / "families.pdf"
META_SOURCE = DATA / "orbit_metadata.json"
AMPLITUDE_SOURCE = DATA / "amplitude_scan.csv"
metadata = json.loads(META_SOURCE.read_text())
ALPHA = metadata["alpha"]
ETA = metadata["eta"]
B_CRIT = metadata["Bcrit"]
with AMPLITUDE_SOURCE.open(newline="") as fp:
    seed = next(row for row in csv.DictReader(fp) if abs(float(row["alpha"]) - ALPHA) < 1e-9)
EPSILON = float(seed["epsilon"])
assert abs(EPSILON - 0.2) < 1e-12
X_CRIT = B_CRIT / np.sqrt(2)

BLUE = "#0072B2"
ORANGE = "#C46800"
BLACK = "#242424"
GRAY = "#737373"
PALE = "#EDEDED"
DASH = (0, (5, 2.5))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": .65,
    "xtick.major.width": .65,
    "ytick.major.width": .65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
})

fig = plt.figure(figsize=(6.8, 4.3), facecolor="white")
fig.text(.055, .964, "(a)  Known families at fixed charges and coupling",
         fontsize=10.5, fontweight="bold", va="top")
fig.text(.115, .908,
         r"$Q_1=Q_2=1,\quad \alpha=4.893760362084\ldots$",
         fontsize=9)

ax = fig.add_axes([.115, .515, .84, .345])
ax.set_xlim(.6, 6.6)
ax.set_ylim(-.32, .32)
ax.set_xlabel(r"$B/\sqrt{V_{\rm eff}^{0}}=B/\sqrt{2}$", labelpad=6)
ax.set_ylabel(r"Horizon scalar $\phi_+$", labelpad=8)
ax.set_xticks([1, 2, X_CRIT, 4, 5, 6])
ax.set_xticklabels(["1", "2", r"$x_{\rm crit}$", "4", "5", "6"])
ax.set_yticks([-.2, 0, .2])
ax.set_yticklabels([r"$-0.2$", "0", r"$+0.2$"])
ax.tick_params(direction="out")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.axvline(X_CRIT, color="#B5B5B5", lw=.8, ls=(0, (2, 3)),
           ymin=.06, ymax=.94, zorder=1)


def ray(axis, start, end, y, color, linestyle="solid", direction=1):
    """Draw a ray with an arrowhead while retaining the line style."""
    axis.plot([start, end], [y, y], color=color, lw=1.8,
              ls=linestyle, solid_capstyle="butt", zorder=3)
    # The arrowhead is solid so both solid and dashed rays remain clear.
    tip, tail = (end, end-.13) if direction > 0 else (start, start+.026)
    axis.add_patch(FancyArrowPatch((tail, y), (tip, y),
                                  arrowstyle="-|>", mutation_scale=9,
                                  color=color, lw=1.2, zorder=4))


ray(ax, 1, 6.44, 0, BLACK)
ray(ax, X_CRIT, 6.44, EPSILON, BLUE)
ray(ax, X_CRIT, 6.44, -EPSILON, ORANGE, DASH)
ax.plot(1, 0, "o", ms=5.8, color=BLACK, zorder=5)
for y, color in [(EPSILON, BLUE), (-EPSILON, ORANGE)]:
    ax.plot(X_CRIT, y, "o", ms=6.4, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.5, zorder=6)

ax.text(3.16, .244, r"Hairy $+$", color=BLUE, fontsize=9.5)
ax.text(3.16, -.271, r"Hairy $-$", color=ORANGE, fontsize=9.5)
ax.text(1.16, .043, r"Constant scalar Reissner–Nordström: $\eta=1$", fontsize=9.2)
ax.text(1.08, -.063, "Extremal Reissner–Nordström", fontsize=8, color=GRAY)
ax.text(4.73, .110, r"Hairy pair: $\eta\simeq0.999859966474$",
        ha="center", fontsize=8.6)
ax.text(4.75, -.108, "Identical metrics under scalar reflection",
        ha="center", fontsize=8.2, color=GRAY)
ax.text(X_CRIT-.12, .272, r"$x_{\rm crit}\simeq2.973$",
        ha="right", fontsize=8.3, color=GRAY)

fig.text(.055, .379, "(b)  Asymptotic modulus of the same families",
         fontsize=10.5, fontweight="bold", va="top")
ax2 = fig.add_axes([.115, .090, .84, .212])
ax2.set_xlim(-.63, .63)
ax2.set_ylim(-.40, .36)
ax2.set_axis_off()

# The grey segment is only a guide for the modulus coordinate. It is not a
# solution branch; its two intervals are expressly unclassified below.
ax2.plot([-.2, .2], [0, 0], color="#B9B9B9", lw=1.1, zorder=1)
ax2.axvspan(-.2, .2, ymin=.43, ymax=.62, color=PALE, zorder=0)
ax2.plot([-.583, -.2], [0, 0], color=ORANGE, lw=1.8, ls=DASH, zorder=3)
ax2.plot([.2, .583], [0, 0], color=BLUE, lw=1.8, zorder=3)
for sign, color in [(-1, ORANGE), (1, BLUE)]:
    ax2.add_patch(FancyArrowPatch((sign*.55, 0), (sign*.596, 0),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=1.3, zorder=4))
    ax2.plot(sign*EPSILON, 0, "o", ms=6.4, markerfacecolor="white",
             markeredgecolor=color, markeredgewidth=1.5, zorder=6)
    ax2.text(sign*.392, .105,
             r"Hairy $-$" if sign < 0 else r"Hairy $+$",
             color=color, ha="center", va="bottom", fontsize=9.5)
    ax2.text(sign*.2, -.078, r"$-0.2$" if sign < 0 else r"$+0.2$",
             ha="center", va="top", fontsize=9)
    ax2.text(sign*.2, -.245, r"$B\to\infty$", ha="center", va="top",
             fontsize=8, color=GRAY)
    ax2.text(sign*.579, -.078, r"$-\infty$" if sign < 0 else r"$+\infty$",
             ha="center", va="top", fontsize=9)
    ax2.text(sign*.442, -.245, r"$B\to B_{\rm crit}^{+}$", ha="center",
             va="top", fontsize=8, color=GRAY)
ax2.plot(0, 0, "o", ms=5.8, color=BLACK, zorder=6)
ax2.text(0, .105, "Reissner–Nordström", ha="center", va="bottom", fontsize=8.2)
ax2.text(0, -.078, "0", ha="center", va="top", fontsize=9)
ax2.text(.63, .07, r"$\phi_\infty$", ha="right", va="bottom", fontsize=10)
fig.text(.535, .043,
         "Grey intervals: not covered by the families shown.",
         ha="center", fontsize=8.2, color=GRAY)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, metadata={
    "Title": "Known fixed-charge families and asymptotic modulus domains",
    "Subject": "Schematic of the Reissner–Nordström and reflection-related tuned hairy orbits",
    "Creator": "Matplotlib; plot_families.py",
    "CreationDate": None,
    "ModDate": None,
})
plt.close(fig)
print(f"Wrote {OUTPUT}")
print(f"Bcrit/sqrt(2) = {X_CRIT:.12f}")

