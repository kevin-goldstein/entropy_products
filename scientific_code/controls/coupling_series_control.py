"""Exact coupling-law and low-order tuned-series checks using only SymPy.

Imports none of the supplied recurrence algorithms.  Assertion failures exit
with nonzero status.  Run with Python and the package's existing SymPy dependency.
"""
import sympy as s

# Differentiate the four-dimensional matter Lagrangian at fixed field strengths
# before substituting Maxwell's solution.  f, theta and their coupling jets are
# independent symbols, so this includes simultaneous electric/magnetic charges.
f, theta, fm, tm, gm, vm, Q, P, b, A, phip = s.symbols(
    'f theta fm tm gm vm Q P b A phip', nonzero=True)
E = (Q+theta*P)/(f*b**2)
W = (Q+theta*P)**2/f+f*P**2
Wm = s.diff(W,f)*fm+s.diff(W,theta)*tm
matter_derivative = b**2/s.Integer(4)*(
    -2*gm*A*phip**2-fm*(-2*E**2+2*P**2/b**4)
    -2*vm-4*tm*E*P/b**2)
assert s.simplify(matter_derivative+(Wm/b**2+b**2*vm+A*b**2*gm*phip**2)/2)==0
print('PASS: fixed-field Lagrangian coupling derivative, including electric, magnetic and axionic terms')
rp,rm=s.symbols('rp rm',positive=True)
rn_integral=(1/rm-1/rp)/(rp-rm)
assert s.simplify(2*rn_integral-2/(rp*rm))==0
print('PASS: RN product normalization d log(C)/d mu=2/mu for Veff=mu=rp*rm')

z = s.symbols('z')
p = s.legendre(3, z)
integral = lambda f: s.integrate(s.expand(f), (z, -1, 1))
L = lambda f: s.expand(s.diff((z*z-1)*s.diff(f,z),z)-12*f)
beta = (9-9*z**2+15*z**4-15*z**6)/8
Lambda2 = -integral(p*(48*p**3-24*beta*p))/integral(p*p)

# Direct monomial linear solve for the cubic scalar response.
a = s.symbols('a0:5')
q = sum(a[i]*z**(2*i+1) for i in range(5))
conditions = s.Poly(L(q)-(Lambda2*p+48*p**3-24*beta*p),z).coeffs()
q = s.expand(q.subs(s.solve(conditions+[q.subs(z,1)], a, dict=True)[0]))
assert L(q) == s.expand(Lambda2*p+48*p**3-24*beta*p)

# Integrate the fourth-order metric response with the horizon constraint.
gamma_pp = -beta*s.diff(p,z)**2-2*s.diff(p,z)*s.diff(q,z)
gamma_p = s.integrate(gamma_pp,z)
gamma_p = gamma_p-gamma_p.subs(z,1)-(Lambda2+24)/2
gamma = s.integrate(gamma_p,z)
gamma = s.expand(gamma-gamma.subs(z,1))

# Order-five scalar solvability determines Lambda4 without finding phi_5.
source5 = s.expand(
    Lambda2*q+144*p*p*q+8*Lambda2*p**3+s.Rational(288,5)*p**5
    -2*beta*(12*q+Lambda2*p+48*p**3)
    +12*p*(3*beta**2-2*gamma))
Lambda4 = -integral(p*source5)/integral(p*p)
d2 = -Lambda2/7
d4 = (d2*d2-Lambda4)/7

# Independently obtain the entropy through order six from the coupling law.
# d log(eta)/d epsilon = Lambda'(epsilon) integral d_Lambda(cosh)/h^2 dz.
I2 = integral(p*p)
I4 = integral(2*p*q+4*p**4-2*beta*p*p)
sigma4 = Lambda2*I2/2
sigma6 = (Lambda2*I4+2*Lambda4*I2)/3

for name in ('Lambda2','Lambda4','d2','d4','I2','I4','sigma4','sigma6'):
    print(name, '=', globals()[name])
print('q =', q)
print('gamma(-1) =',gamma.subs(z,-1))
print('coefficient of [(1-eta)-(3-nu)epsilon^2]/epsilon^6 =',-sigma6-d4)
assert d2 == s.Rational(96,1001)
assert d4 == -s.Rational(38719078400,323969969323)
assert sigma4 == -s.Rational(96,1001)
assert sigma6 == s.Rational(51807744,231175945)
assert -sigma6-d4 == -s.Rational(169421470208,1619849846615)
print('PASS: all exact low-order coefficients independently reproduced')
