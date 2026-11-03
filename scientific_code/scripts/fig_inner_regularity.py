r"""
fig_inner_regularity.py -- supplementary diagnostic figure: |(r - r_-) phi'(r)| approaching the inner horizon, for
V_eff = e^{alpha1 phi} + e^{alpha2 phi}, starting from PHYSICAL outer-horizon data (b'(r+) > 0, asymptotically
flat exterior, checked in inner_regularity_physical.py), r+ = 1, r- = 0.35.
Toda couplings and a tuned non-Toda branch: -> 0 linearly (regular inner horizon).  Generic couplings, with integer
nu or not: plateau (log singularity).
Usage: python3 fig_inner_regularity.py [output directory]   (default: results/data/figures in the package)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inner_regularity_physical as ip
import checklib as ck

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.environ.get("HOT_ATTRACTORS_OUTPUT", os.path.join(HERE, "..", "results", "data")), "figures")
os.makedirs(OUTDIR, exist_ok=True)


def profile(a1, a2, kb, dphi, expect):
    ps, bde, Vs = ip.doubly_extremal(a1, a2)
    bp, php = kb * bde, ps + dphi
    o = ip.run(a1, a2, bp, php)
    ck.true('(%.4f,%.4f) b+/bDE=%.1f dphi=%+.2f: physical data, asymptotically flat exterior' % (a1, a2, kb, dphi),
            o['physical'] and o['flat'])
    dg = ip.diagnose(a1, a2, bp, php)
    ck.true('(%.4f,%.4f): inner-horizon outcome is %s' % (a1, a2, expect), dg['outcome'] == expect, ip.fmt_c(dg))
    sol = ip.inward(a1, a2, bp, php)
    eps = np.logspace(-1, -8, 200)
    values = ip.sample_inward(sol, ip.RM + eps, 'inner-regularity figure profile')
    return eps, np.abs(eps * values[3])


at = ip.tuned_alpha(0.2)          # Davydov-type tuned branch at b+ = 2.5 b+^DE, eps = 0.2 (nu = 2.99635)
models = [((2.0, -2.0), 1.3, 0.2, r'$(2,-2)$: $A_1\oplus A_1$ Toda, $\nu=1$', '-', 'zero'),
          ((2 * np.sqrt(3), -2 * np.sqrt(3)), 1.3, 0.1, r'$(2\sqrt{3},-2\sqrt{3})$: $A_2$ Toda, $\nu=2$', '--', 'zero'),
          ((at, -at), 2.5, 0.2, r'$(4.8938,-4.8938)$: tuned branch, $\nu\simeq2.9963$', (0, (1, 1)), 'zero'),
          ((2.0, -6.0), 1.3, 0.2, r'$(2,-6)$: not Toda, $\nu=2$', (0, (5, 1, 1, 1, 1, 1)), 'nonzero'),
          ((2.0, -1.0), 1.3, 0.2, r'$(2,-1)$: $\nu\simeq0.618$', '-.', 'nonzero'),
          ((1.0, -1.0), 1.3, 0.05, r'$(1,-1)$: $\nu\simeq0.366$', ':', 'nonzero')]
colors = ['#2a78d6', '#eb6834', '#7a52c7', '#1baf7a', '#eda100', '#e87ba4']
short = [r'$\nu=1$', r'$\nu=2$', r'tuned', r'$\nu=2$ (non-Toda)', r'$\nu\simeq0.618$', r'$\nu\simeq0.366$']
offsets = [-6, 6, 0, 0, 0, 0]
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
fig, ax = plt.subplots(figsize=(5.6, 3.5))
for ((a1, a2), kb, dphi, lab, ls, expect), col, dy, sh in zip(models, colors, offsets, short):
    eps, vals = profile(a1, a2, kb, dphi, expect)
    ax.loglog(eps, vals, linestyle=ls, color=col, lw=1.6, label=lab)
    ax.annotate(sh, xy=(eps[-1], vals[-1]), xytext=(4, dy), textcoords='offset points', va='center', fontsize=8,
                color='#0b0b0b')
ax.set_xlabel(r'$r-r_-$')
ax.set_ylabel(r'$|(r-r_-)\,\phi^\prime(r)|$')
ax.grid(True, which='major', color='#e2e1dc', lw=0.5)
ax.set_xlim(1e-1, 1e-9)
ax.legend(frameon=False, fontsize=7.5, loc='lower left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext, kw in (('pdf', {}), ('png', {'dpi': 160})):
    path = os.path.join(OUTDIR, 'inner_regularity.' + ext)
    fig.savefig(path, **kw)
    ck.true('figure written to %s' % os.path.relpath(path, HERE), os.path.getsize(path) > 1000)
ck.finish()
