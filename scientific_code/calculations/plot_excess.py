#!/usr/bin/env python3
"""Figure for the explicit excess example: (a) scalar profile against areal radius, (b) eta along the
two-node branch of the slope family (s, -2s). Needs excess_example.py in the same directory."""
from pathlib import Path
import os, sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controls'))
import asymmetric_excess as ex
OUT=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',ROOT/'results/data'))
DATA=OUT/'asymmetric'; DATA.mkdir(parents=True,exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import brentq

# ---------- data: profile of the explicit solution
th = ex.Theory(np.sqrt(7), -2*np.sqrt(7), 2.0, 1.0)
rp, rm = 1.0, 0.35; d = rp - rm
bDE = (d + np.sqrt(d*d + 4*th.V0))/2; B = 2.5*bDE
eps = 0.4427936622730
x, so, si, res, dbp = ex.two_sided(th, B, [eps, np.sqrt(1.0220531880189)*3/B, eps], dense=True)
eps, bm, pm = x
ext = ex.exterior(th, B, eps)
r_in = np.linspace(rm + 1e-4, 0.5*(rp + rm), 4000)
r_out = np.linspace(0.5*(rp + rm), rp - 1e-3, 4000)
r_ext = np.geomspace(rp + 1e-3, 4e5, 6000)
b_int = np.concatenate([si.sol(r_in)[0], so.sol(r_out)[0]]); p_int = np.concatenate([si.sol(r_in)[2], so.sol(r_out)[2]])
ye = ext['sol'].sol(r_ext); b_ext, p_ext = ye[0], ye[2]
# prepend exact horizon values
b_int = np.concatenate([[bm], b_int, [B]]); p_int = np.concatenate([[pm], p_int, [eps]])
np.savetxt(DATA/'profile_interior.csv', np.c_[np.concatenate([[rm], r_in, r_out, [rp]]), b_int, p_int],
           delimiter=',', header='r,b,phi', comments='')
np.savetxt(DATA/'profile_exterior.csv', np.c_[r_ext, b_ext, p_ext], delimiter=',',
           header='r,b,phi', comments='')
zeros = [0.5*(b_int[i] + b_int[i+1]) for i in range(len(p_int) - 1) if np.sign(p_int[i]) != np.sign(p_int[i+1])]
print('interior zeros of phi at b =', zeros, ' min phi =', p_int.min())

# ---------- data: two-node branch in the family a1 = s, a2 = -2 s (Q^2 = 2, P^2 = 1)
def branch(sign):
    out = []; s_prev = np.sqrt(6.0)
    for e in sign*np.concatenate([np.geomspace(0.005, 0.2, 12), np.linspace(0.25, 3.0, 23)]):
        g = lambda s: ex.even_mismatch(ex.Theory(s, -2*s, 2.0, 1.0), e)
        for w in (0.002, 0.005, 0.01, 0.02, 0.05):
            grid = s_prev*np.linspace(1 - w, 1 + w, 21); vals = [g(t) for t in grid]
            br = [(grid[i], grid[i+1]) for i in range(20) if np.sign(vals[i]) != np.sign(vals[i+1])]
            if br:
                lo, hi = min(br, key=lambda q: abs(0.5*(q[0] + q[1]) - s_prev)); break
        if not br:
            raise RuntimeError("No continuation bracket found")
        s = brentq(g, lo, hi, xtol=1e-14); s_prev = s
        t = ex.Theory(s, -2*s, 2.0, 1.0)
        out.append((e, t.nu(), ex.eta_autonomous(t, e)))
    return np.array(out)
bp_, bm_ = branch(+1), branch(-1)
br = np.vstack([bm_[::-1], [[0.0, 2.0, 1.0]], bp_])
np.savetxt(DATA/'branch_r2_n2.csv', br, delimiter=',', header='phi_plus,nu,eta', comments='')
print('branch nu range', br[:, 1].min(), br[:, 1].max(), ' eta range', br[:, 2].min(), br[:, 2].max())

# ---------- figure
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'font.size': 8.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE = '#2a78d6'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw=dict(wspace=0.32))

# (a) scalar profile against areal radius
bb = np.concatenate([b_int, b_ext]); pp = np.concatenate([p_int, p_ext])
ax1.axhline(0.0, color=GRID, lw=0.8, zorder=0)
ax1.axhline(ext['phinf'], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
for bh in (bm, B):
    ax1.axvline(bh, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
ax1.plot(bb, pp, color=BLUE, lw=1.6, zorder=3)
ax1.set_xscale('log'); ax1.set_xlim(0.45, 4e4); ax1.set_ylim(-0.32, 1.08)
ax1.set_xlabel('areal radius $b$'); ax1.set_ylabel(r'scalar $\phi$')
ax1.text(bm*1.08, -0.29, r'$b_-$', color=INK2, va='bottom'); ax1.text(B*1.08, -0.29, r'$b_+$', color=INK2, va='bottom')
ax1.text(3e4, ext['phinf'] - 0.03, r'$\phi_\infty=%.3f$' % ext['phinf'], color=INK2, ha='right', va='top')
ax1.text(30, 0.47, r'$\phi_-=\phi_+=%.4f$' % eps + '\n(both horizons)', color=INK, fontsize=8, va='center')
ax1.text(-0.16, 1.04, '(a)', transform=ax1.transAxes, ha='left', va='bottom', fontsize=9, color=INK)
for s_ in ('top', 'right'): ax1.spines[s_].set_visible(False)

# (b) entropy product along the two-node branch
ax2.axhline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax2.plot(br[:, 1], br[:, 2], color=BLUE, lw=1.6, zorder=3)
nu_ex = th.nu(); eta_ex = 1.0220531880189
ax2.plot([nu_ex], [eta_ex], 'o', ms=6.5, mfc=BLUE, mec='white', mew=1.2, zorder=4)
ax2.annotate(r'$\nu=2.1926$, $\eta=1.0221$', (nu_ex, eta_ex), xytext=(1.80, 1.12), fontsize=8, color=INK,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax2.text(2.36, 1.205, r'$\phi_+>0$: excess', color=INK, ha='right', va='bottom', fontsize=8)
ax2.text(1.80, 0.905, r'$\phi_+<0$: deficit', color=INK, ha='left', va='bottom', fontsize=8)
ax2.text(2.385, 0.992, 'RN, $\\eta=1$', color=INK2, ha='right', va='top', fontsize=8)
ax2.plot([2.0], [1.0], 'o', ms=4, mfc='white', mec=INK2, mew=0.8, zorder=4)
ax2.set_xlabel(r'$\nu$'); ax2.set_ylabel(r'$\eta=S_+S_-/S_{\rm ext}^2$')
ax2.set_xlim(1.76, 2.40); ax2.set_ylim(0.90, 1.24)
ax2.text(-0.2, 1.04, '(b)', transform=ax2.transAxes, ha='left', va='bottom', fontsize=9, color=INK)
for s_ in ('top', 'right'): ax2.spines[s_].set_visible(False)
for ax in (ax1, ax2):
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
fig.savefig(FIG/'excess_example.pdf', bbox_inches='tight'); fig.savefig(FIG/'excess_example.png', dpi=200, bbox_inches='tight')
print('[PASS] Generated excess profile and coupling-product figure')
