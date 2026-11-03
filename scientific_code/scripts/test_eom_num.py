import sympy as sp, mpmath as mp
import checklib as ck
from eom_num import make_residuals, r, th
mp.mp.dps = 25
# Kerr-Newman
M, a, Q = sp.symbols('M a Q', positive=True)
rho2 = r**2 + a**2*sp.cos(th)**2; Dl = r**2 - 2*M*r + a**2 + Q**2
g = sp.zeros(4,4)
g[0,0] = -(Dl - a**2*sp.sin(th)**2)/rho2
g[0,3] = g[3,0] = -a*sp.sin(th)**2*(r**2 + a**2 - Dl)/rho2
g[3,3] = ((r**2 + a**2)**2 - Dl*a**2*sp.sin(th)**2)/rho2*sp.sin(th)**2
g[1,1] = rho2/Dl; g[2,2] = rho2
A = [-Q*r/rho2, sp.Integer(0), sp.Integer(0), Q*r*a*sp.sin(th)**2/rho2]
res = make_residuals(g, A, [], lambda s: sp.zeros(0,0), lambda s: sp.Integer(1), lambda s: sp.Integer(0), lambda s: sp.Integer(0), (M,a,Q))
E, Mx, S = res(mp.mpf('3.1'), mp.mpf('0.7'), (mp.mpf('1.3'), mp.mpf('0.6'), mp.mpf('0.7')))
print('KN: max |Einstein| =', mp.nstr(ck.max_abs(E), 3), ' Maxwell =', [mp.nstr(x,3) for x in Mx])
ck.small('Kerr-Newman: Einstein residuals', ck.max_abs(E), 1e-20)
ck.small('Kerr-Newman: Maxwell residuals', ck.max_abs(Mx), 1e-20)
# negative control: rescale the gauge potential by 1.1 at fixed metric
A11 = [sp.Rational(11, 10) * x for x in A]
res11 = make_residuals(g, A11, [], lambda s: sp.zeros(0,0), lambda s: sp.Integer(1), lambda s: sp.Integer(0), lambda s: sp.Integer(0), (M,a,Q))
E11, Mx11, S11 = res11(mp.mpf('3.1'), mp.mpf('0.7'), (mp.mpf('1.3'), mp.mpf('0.6'), mp.mpf('0.7')))
ck.large('negative control: A -> 1.1 A at fixed metric violates the Einstein equations', ck.max_abs(E11), 1e-3)
ck.finish()
