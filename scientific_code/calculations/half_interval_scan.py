"""Reversible n=3 branch: independent half-interval shooting and amplitude scan.

Requires numpy/scipy. The endpoint series is generated here, rather than importing
either of the manuscript's collocation/two-sided shooting implementations.
"""
import csv
import json
from pathlib import Path
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

OUT = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"branch"
RM, RP = 0.35, 1.0
D = RP - RM
BDE = (D + np.sqrt(D * D + 8.0)) / 2.0
B = 2.5 * BDE


def product(a, b, order):
    return np.convolve(a, b)[:order + 1]


def inverse(a, order):
    b = np.zeros(order + 1)
    b[0] = 1.0 / a[0]
    for n in range(1, order + 1):
        b[n] = -sum(a[k] * b[n-k] for k in range(1, n+1)) / a[0]
    return b


def exponential(a, order):
    b = np.zeros(order + 1)
    b[0] = np.exp(a[0])
    for n in range(1, order+1):
        b[n] = sum(k * a[k] * b[n-k] for k in range(1, n+1)) / n
    return b


def regular_series(rh, bh, ph, alpha, order):
    """Ordinary power-series coefficients, including factorial denominators."""
    dh = 2*rh - RM - RP
    b = np.zeros(order+1)
    phi = np.zeros(order+1)
    b[0], phi[0] = bh, ph
    b[1] = (bh - 2*np.cosh(alpha*ph)/bh) / dh
    for n in range(order):
        invb = inverse(b, order)
        sinh = (exponential(alpha*phi, order) - exponential(-alpha*phi, order))/2
        force = alpha*product(sinh, product(invb, invb, order), order)[n]
        phi[n+1] = (force - n*(n+1)*phi[n]) / (dh*(n+1)**2)
        if n+2 <= order:
            dphi = np.r_[np.arange(1, order+1)*phi[1:], 0.0]
            b[n+2] = -product(b, product(dphi, dphi, order), order)[n]/((n+1)*(n+2))
    return b, phi


def initial(rh, bh, ph, alpha, cutoff, order):
    b, phi = regular_series(rh, bh, ph, alpha, order)
    s = -cutoff if rh == RP else cutoff
    val = np.polynomial.polynomial.polyval
    return [val(s, b), val(s, np.arange(1, order+1)*b[1:]),
            val(s, phi), val(s, np.arange(1, order+1)*phi[1:])]


def rhs(r, y, alpha):
    b, db, phi, dphi = y
    p = (r-RM)*(r-RP)
    dp = 2*r-RM-RP
    return [db, -b*dphi*dphi, dphi,
            (alpha*np.sinh(alpha*phi)/(b*b)-dp*dphi)/p]


def u_dot(r, y):
    p = (r-RM)*(r-RP)
    return (2*r-RM-RP)/2-p*y[1]/y[0]


def half(alpha, epsilon, cutoff=1e-6, order=5, rtol=3e-14):
    def event(r, y):
        return u_dot(r, y)
    event.terminal = True
    sol = solve_ivp(lambda r,y: rhs(r,y,alpha), (RP-cutoff, RM+cutoff),
                    initial(RP, B, epsilon, alpha, cutoff, order),
                    events=event, dense_output=True, method='DOP853',
                    rtol=rtol, atol=rtol/100)
    if not sol.success or len(sol.t_events[0]) != 1 or not np.all(np.isfinite(sol.y)) or np.min(sol.y[0]) <= 0:
        raise RuntimeError(sol.message)
    return sol, sol.t_events[0][0], sol.y_events[0][0]


def tuned(epsilon, cutoff=1e-6, order=5, rtol=3e-14):
    seed = np.sqrt(24-192*epsilon*epsilon/143)
    fun = lambda alpha: half(alpha, epsilon, cutoff, order, rtol)[2][2]
    alpha = brentq(fun, seed-0.015, seed+0.015, xtol=8e-15)
    sol, rc, yc = half(alpha, epsilon, cutoff, order, rtol)
    kc = (RP-rc)/(rc-RM)
    bm = B/kc
    eta = (B*bm/2)**2
    nu = (np.sqrt(1+2*alpha*alpha)-1)/2
    # A separate regular integration from the reflected inner horizon checks
    # the four central Cauchy data, rather than merely reflecting the array.
    si = solve_ivp(lambda r,y: rhs(r,y,alpha), (RM+cutoff, rc),
                   initial(RM, bm, -epsilon, alpha, cutoff, order),
                   dense_output=True, method='DOP853', rtol=rtol, atol=rtol/100)
    if not si.success or abs(si.t[-1]-rc) > 1e-12 or not np.all(np.isfinite(si.y)) or np.min(si.y[0]) <= 0:
        raise RuntimeError('Inner integration did not reach the center regularly')
    central_mismatch = float(np.max(np.abs(si.y[:,-1]-yc)))
    # Evaluate both independent integrations at reflected radius pairs.
    ss = np.linspace(0.0, 3.0, 101)
    kin, kout = kc*np.exp(ss), kc*np.exp(-ss)
    rin = (RP+kin*RM)/(1+kin)
    rout = (RP+kout*RM)/(1+kout)
    if rin.min() < si.t.min() or rin.max() > si.t.max()+1e-14 or rout.min() < sol.t.min()-1e-14 or rout.max() > sol.t.max():
        raise RuntimeError('Reflected samples fall outside integrated intervals')
    yin, yout = si.sol(rin), sol.sol(rout)
    ain = (rin-RM)*(rin-RP)/yin[0]**2
    aout = (rout-RM)*(rout-RP)/yout[0]**2
    row = dict(epsilon=epsilon, alpha=alpha, nu=nu, delta_nu=3-nu,
               eta=eta, deficit=1-eta, deficit_over_epsilon4=(1-eta)/epsilon**4,
               delta_nu_over_epsilon2=(3-nu)/epsilon**2,
               bplus=B, bminus=bm, center_radius=rc,
               central_phi=float(yc[2]), center_match=central_mismatch,
               symmetry_phi=float(np.max(np.abs(yin[2]+yout[2]))),
               symmetry_a2=float(np.max(np.abs(ain-aout))),
               cutoff=cutoff, order=order, rtol=rtol)
    if not all(np.isfinite(v) for v in row.values()):
        raise RuntimeError('Nonfinite half-interval observable')
    if max(row['center_match'], row['symmetry_phi'], row['symmetry_a2']) > 1e-8:
        raise RuntimeError('Independent center/symmetry check failed')
    return row


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [tuned(e) for e in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]]
    for row in rows:
        print(json.dumps(row), flush=True)
    with (OUT/'amplitude_scan.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = [tuned(0.2, cutoff=cutoff, order=order, rtol=rtol)
                  for cutoff,order,rtol in [(1e-5,4,1e-12),(1e-6,5,2e-13),(1e-7,5,3e-14)]]
    (OUT/'half_interval_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
