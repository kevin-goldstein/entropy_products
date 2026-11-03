"""Vector schematic of known families in the fixed symmetric theory.

This is a domain diagram, not a numerical continuation or an exhaustive phase
diagram. All finite coordinates in panel (a) are the values stated in the
manuscript. Panel (b) is a number-line domain schematic: arrows denote
unbounded modulus intervals, not finite endpoints of a plotted data range.
No ODE integration, fitting, or new physical computation is performed.
"""
from pathlib import Path
import argparse
import os
import csv
import json
import hashlib
import tempfile
from output_paths import path_label

_cache = tempfile.TemporaryDirectory(prefix="hot-attractors-families-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name)
os.environ.setdefault("XDG_CACHE_HOME", _cache.name)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", ROOT / "results/data"))

def source(relative):
    fresh = OUT / relative
    return fresh if fresh.is_file() else ROOT / "data/generated" / relative
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path,
                    default=OUT / "figures" / "families.pdf")
OUTPUT = parser.parse_args().output.resolve()

META_SOURCE = source("orbit/orbit_metadata.json")
AMPLITUDE_SOURCE = source("branch/amplitude_scan.csv")
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
ax.text(1.16, .043, r"Constant scalar RN: $\eta=1$", fontsize=9.2)
ax.text(1.08, -.063, "Extremal RN", fontsize=8, color=GRAY)
ax.text(4.73, .087, r"Hairy pair: $\eta\simeq0.999859966474$",
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
ax2.text(0, .105, "RN", ha="center", va="bottom", fontsize=9.2)
ax2.text(0, -.078, "0", ha="center", va="top", fontsize=9)
ax2.text(.63, .07, r"$\phi_\infty$", ha="right", va="bottom", fontsize=10)
fig.text(.535, .043,
         "Grey intervals: not covered by the families shown.",
         ha="center", fontsize=8.2, color=GRAY)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, metadata={
    "Title": "Known fixed-charge families and asymptotic modulus domains",
    "Subject": "Schematic of the RN and reflection-related tuned hairy orbits",
    "Creator": "Matplotlib; plot_families.py",
    "CreationDate": None,
    "ModDate": None,
})
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
plt.close(fig)
print(f"Wrote {OUTPUT}")
print(f"Bcrit/sqrt(2) = {X_CRIT:.12f}")

record = {"schematic": True, "new_ODE_solves": False, "alpha": ALPHA, "eta": ETA, "epsilon": EPSILON, "Bcrit": B_CRIT, "sources": [{"path": path_label(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (META_SOURCE, AMPLITUDE_SOURCE)]}
(OUTPUT.parent / "families_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
