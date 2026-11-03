r"""
ddim_identity.py -- in D dimensions the combination R^t_t + (D-3) R^th_th is free of scalar-kinetic and
Maxwell terms; in the variable rho with d rho/dr = (D-3) B^{D-4} it reads
    d^2/drho^2 ( A B^{2(D-3)} ) = 2 - 4 B^2 V / (D-3)^2          (1811.04963, eq. 3.11)
We verify this on-shell, and hence  T_pm S_pm = +- (D-3) Omega_{D-2} (rho_+ - rho_-)/(16 pi)  for V = 0.
"""
import sympy as sp
import checklib as ck
from check_K_conservation import field_equations, A, B, ph, r, V

for D in (4, 5, 6):
    eqs, E = field_equations(D)
    P = A * B ** (2 * (D - 3))
    drho = lambda X: sp.diff(X, r) / ((D - 3) * B ** (D - 4))
    lhs = drho(drho(P)) - 2 + 4 * B ** 2 * V(ph) / (D - 3) ** 2
    # express lhs as a combination of the mixed Einstein equations E^t_t and E^th_th
    a1, a2 = sp.symbols('a1 a2')
    comb = sp.simplify(lhs - (a1 * eqs[(0, 0)] + a2 * eqs[(2, 2)]))
    sol = sp.solve(sp.Poly(sp.expand(sp.numer(sp.together(comb))), sp.diff(A, r, 2), sp.diff(B, r, 2)).coeffs(), [a1, a2], dict=True)
    resid = sp.simplify(comb.subs(sol[0])) if sol else None
    print('D=%d:' % D, sol, '  residual ->', resid)
    ck.true('D=%d: the rho-identity is a combination of E^t_t and E^th_th' % D, sol and resid == 0)
ck.finish()
