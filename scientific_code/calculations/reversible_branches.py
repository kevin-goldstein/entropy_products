"""Scaled autonomous shooting for the symmetric two-exponential model.

The integration starts at the outer horizon asymptote and stops at the unique
maximum of the metric variable. Reflection completes a two-horizon solution
when phi=0 (odd node count) or phi_dot=0 (even node count) at that maximum.
No conclusion about nonreversible solutions follows from this matching test.

In the manuscript's autonomous equations U_tau_tau=-exp(2U)*V_eff, set
t=(d/2)*tau plus a translation, w=U-log(d/2)+log(V_eff(epsilon))/2, and
h=log(cosh(alpha*epsilon)). The translation fixes w~t at the outer endpoint
t -> -infinity. Then E=exp(2w)*cosh(alpha*phi)/cosh(alpha*epsilon), and the
energy is w_dot**2 + phi_dot**2 + E = 1. Reflection gives w~-t+2*t_center
at the other endpoint, yielding log(eta)=2*(log(4)+h-2*t_center).
This scaling keeps the potential exponential bounded through epsilon=6
without subtracting two huge terms. The equations are integrated independently
of the radial shooting and collocation implementations elsewhere in the package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class Settings:
    cutoff: float = 18.0
    rtol: float = 2e-12
    max_step: float = 0.15

    def metadata(self):
        return asdict(self) | {"method": "DOP853", "atol": self.rtol / 100}


def logcosh(value):
    return np.logaddexp(value, -value) - math.log(2)


def shoot(nu: float, epsilon: float, settings: Settings = Settings()):
    """Return the outer-to-center solve, including all scalar zero events."""
    if not (math.isfinite(nu) and nu > 0 and math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("nu and epsilon must be finite and strictly positive")
    alpha = math.sqrt(2 * nu * (nu + 1))
    h = float(logcosh(alpha * epsilon))
    q = math.exp(-2 * settings.cutoff)
    scalar_force = alpha * math.tanh(alpha * epsilon)
    initial = [-settings.cutoff - q / 4, 1 - q / 2,
               epsilon - scalar_force * q / 8, -scalar_force * q / 4]

    def rhs(time, state):
        w, velocity, phi, momentum = state
        potential = np.exp(2 * w + logcosh(alpha * phi) - h)
        return [velocity, -potential, momentum,
                -alpha * np.tanh(alpha * phi) * potential / 2]

    def center(time, state):
        return state[1]

    center.terminal, center.direction = True, -1

    def zero(time, state):
        return state[2]

    solution = solve_ivp(rhs, (-settings.cutoff, max(80., alpha * epsilon)),
                         initial, method="DOP853", rtol=settings.rtol,
                         atol=settings.rtol / 100, max_step=settings.max_step,
                         events=(center, zero))
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError("Center was not reached: " + solution.message)
    tc = float(solution.t_events[0][0])
    state = solution.y_events[0][0]
    potential = float(np.exp(2 * state[0] + logcosh(alpha * state[2]) - h))
    energy = float(state[1] ** 2 + state[3] ** 2 + potential)
    log_eta = 2 * (math.log(4) + h - 2 * tc)
    # A zero at the odd-mode center is added separately by reflection. The
    # fixed time exclusion is far smaller than the separation of interior
    # zeros on the sampled branches and is recorded for reproducibility.
    interior_zeros = [float(t) for t in solution.t_events[1] if t < tc - 1e-6]
    return {"nu": float(nu), "alpha": alpha, "epsilon": float(epsilon),
            "center_time": tc, "center_phi": float(state[2]),
            "center_phidot": float(state[3]), "eta": math.exp(log_eta),
            "deficit": -math.expm1(log_eta), "energy_error": energy - 1,
            "half_nodes": len(interior_zeros), "zero_times": interior_zeros,
            "steps": len(solution.t) - 1}


def residual(result, parity):
    # Dividing by epsilon removes the trivial RN root from near-zero scans.
    return result["center_phi" if parity == "odd" else "center_phidot"] / result["epsilon"]


def checked_row(result, n, settings):
    parity = "odd" if n % 2 else "even"
    nodes = 2 * result["half_nodes"] + n % 2
    if nodes != n:
        raise RuntimeError(f"Branch label n={n} disagrees with scalar zero count {nodes}")
    mismatch = residual(result, parity)
    if abs(mismatch) > 5e-8 or abs(result["energy_error"]) > 2e-9:
        raise RuntimeError(f"Unacceptable matching/energy residual: {mismatch}, {result['energy_error']}")
    return {key: value for key, value in result.items() if key != "zero_times"} | {
        "n": n, "parity": parity, "nodes": nodes,
        "matching_residual": mismatch, "cutoff": settings.cutoff, "rtol": settings.rtol}


def trace_point(n: int, epsilon: float, settings: Settings = Settings()):
    """Solve for nu at prescribed amplitude, then verify the full node count.

    The bracket is deliberately local to n=3,...,7, where sampled windows are
    disjoint. This routine is not an all-mode continuation or fold detector.
    """
    if n not in range(3, 8):
        raise ValueError("The validated continuation brackets cover n=3,...,7")
    parity = "odd" if n % 2 else "even"

    def objective(nu):
        return residual(shoot(nu, epsilon, settings), parity)

    nu = brentq(objective, n - 0.49, n - 1e-10, xtol=5e-15, rtol=9e-16)
    return checked_row(shoot(nu, epsilon, settings), n, settings)


def fixed_coupling(nu: float, bracket, n: int, settings: Settings = Settings()):
    """Hold the action coupling fixed and solve for the scalar amplitude."""
    parity = "odd" if n % 2 else "even"
    objective = lambda epsilon: residual(shoot(nu, epsilon, settings), parity)
    epsilon = brentq(objective, *bracket, xtol=5e-13, rtol=1e-14)
    return checked_row(shoot(nu, epsilon, settings), n, settings)
