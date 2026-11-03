"""Exact algebra and examples for the strengthened Cold-from-hot theorem.

The first checks use the fixed-charge Reissner--Nordstrom branch, including its
smooth extremal endpoint. The second set is an abstract thermodynamic negative
control: the two first laws and signed temperatures alone permit a cusp and two
distinct diagonal endpoints. No spacetime realization of that example is claimed.
Also checks the gap derivative and entropy bracket, harmless critical points,
the field-equation gap floor, and positive-coupling distinct physical branches.
The global analytic factorization theorem is proved in the manuscript; these
examples and algebraic checks do not constitute a numerical proof of it.
"""

import sympy as sp

import checklib as ck


# A physical regular branch: r_+ r_- = q^2, M = q cosh(u), q > 0, u > 0.
q, u = sp.symbols('q u', positive=True)
rp, rm = q * sp.exp(u), q * sp.exp(-u)
mass = q * sp.cosh(u)
entropy_plus, entropy_minus = sp.pi * rp**2, sp.pi * rm**2
temperature_plus = (rp - rm) / (4 * sp.pi * rp**2)
temperature_minus = (rm - rp) / (4 * sp.pi * rm**2)
mass_derivative = sp.diff(mass, u)

ck.zero('RN: outer first law at fixed charge',
        (temperature_plus * sp.diff(entropy_plus, u) - mass_derivative).rewrite(sp.exp))
ck.zero('RN: inner first law with signed temperature',
        (temperature_minus * sp.diff(entropy_minus, u) - mass_derivative).rewrite(sp.exp))
ck.true('RN: no interior rank-zero point', mass_derivative.is_positive)
ck.true('RN: outer entropy increases', sp.diff(entropy_plus, u).is_positive)
ck.true('RN: inner entropy decreases', sp.diff(entropy_minus, u).is_negative)
ck.zero('RN: entropy gap derivative = 4 pi q^2 cosh(2u)',
        sp.diff(entropy_plus - entropy_minus, u)
        - 4 * sp.pi * q**2 * sp.cosh(2 * u))
ck.true('RN: entropy gap derivative is strictly positive',
        (4 * sp.pi * q**2 * sp.cosh(2 * u)).is_positive)
ck.zero('RN: regular entropy image is the positive constant-product graph',
        entropy_plus * entropy_minus - sp.pi**2 * q**4)
for tag, entropy, temperature in (
        ('outer', entropy_plus, temperature_plus),
        ('inner', entropy_minus, temperature_minus)):
    ck.zero('RN: %s extremal entropy = pi q^2' % tag,
            sp.limit(entropy, u, 0, dir='+') - sp.pi * q**2)
    ck.zero('RN: %s extremal temperature vanishes' % tag,
            sp.limit(temperature, u, 0, dir='+'))

# Abstract negative control, -1 < t < 1, with C > 2 if positive entropies
# are desired. All data extend smoothly to t = +/-1. The first-law one-form
# and both entropy derivatives vanish at t = 0, which the regular-branch corollary
# explicitly excludes. A zero derivative is not excluded by rank <= 1 alone.
t, C = sp.symbols('t C', real=True)
gap = 1 - t**2
v = sp.Rational(1, 2) + t / 10
entropy_plus = C - t**2 / 2 - t**3 / 15
entropy_minus = C - 1 + t**2 / 2 - t**3 / 15
temperature_plus = gap / v
temperature_minus = gap / (v - 1)
one_form_coefficient = gap * sp.diff(gap, t)

ck.zero('Countermodel: entropy gap = 1 - t^2',
        entropy_plus - entropy_minus - gap)
ck.zero('Countermodel: outer first law',
        temperature_plus * sp.diff(entropy_plus, t) - one_form_coefficient)
ck.zero('Countermodel: inner first law',
        temperature_minus * sp.diff(entropy_minus, t) - one_form_coefficient)

# This substitution covers the full open interval; SymPy can determine the
# signs exactly from rational functions of the positive variable y.
y = sp.symbols('y', positive=True)
interval_map = (y - 1) / (y + 1)
ck.true('Countermodel: positive gap throughout -1 < t < 1',
        sp.factor(gap.subs(t, interval_map)).is_positive)
ck.true('Countermodel: T+ > 0 throughout -1 < t < 1',
        sp.factor(temperature_plus.subs(t, interval_map)).is_positive)
ck.true('Countermodel: T- < 0 throughout -1 < t < 1',
        sp.factor(temperature_minus.subs(t, interval_map)).is_negative)
ck.zero('Countermodel: first-law one-form vanishes at the interior cusp',
        one_form_coefficient.subs(t, 0))
ck.zero('Countermodel: outer entropy derivative vanishes at the cusp',
        sp.diff(entropy_plus, t).subs(t, 0))
ck.zero('Countermodel: inner entropy derivative vanishes at the cusp',
        sp.diff(entropy_minus, t).subs(t, 0))
jet = sp.Matrix([[sp.diff(entropy, t, order).subs(t, 0) for order in (2, 3)]
                 for entropy in (entropy_plus, entropy_minus)])
ck.zero('Countermodel: independent quadratic and cubic jets give a cusp',
        jet.det() - sp.Rational(4, 5))
ck.true('Countermodel: gap reverses direction across the cusp',
        sp.diff(gap, t).subs(t, -sp.Rational(1, 2)) > 0
        and sp.diff(gap, t).subs(t, sp.Rational(1, 2)) < 0)

for endpoint in (-1, 1):
    ck.zero('Countermodel: diagonal entropy limit at t=%+d' % endpoint,
            (entropy_plus - entropy_minus).subs(t, endpoint))
    ck.zero('Countermodel: outer temperature vanishes at t=%+d' % endpoint,
            temperature_plus.subs(t, endpoint))
    ck.zero('Countermodel: inner temperature vanishes at t=%+d' % endpoint,
            temperature_minus.subs(t, endpoint))
ck.large('Negative control: the two finite diagonal entropy limits differ',
         entropy_plus.subs(t, -1) - entropy_plus.subs(t, 1), 0.1)
print('The abstract cusp violates the regular-branch hypotheses; '
      'no black-hole spacetime realization is asserted.')

# The gap identities do not divide by the possibly vanishing one-form.
tp, tm_abs = sp.symbols('tp tm_abs', positive=True)
fraction = tm_abs / (tp + tm_abs)
dx_dg, dy_dg = fraction, -tp / (tp + tm_abs)
ck.zero('Gap: derivatives differ by one', dx_dg - dy_dg - 1)
ck.zero('Gap: signed first laws agree', tp * dx_dg + tm_abs * dy_dg)
ck.zero('Gap: omega/dg coefficient', tp * dx_dg - tp * tm_abs / (tp + tm_abs))
for name, expression in (
        ('positive outer derivative', dx_dg),
        ('outer derivative below one', 1 - dx_dg),
        ('inner derivative above minus one', 1 + dy_dg),
        ('negative inner derivative', -dy_dg),
        ('omega and dg have the same zeros', tp * tm_abs / (tp + tm_abs))):
    ck.true('Gap: ' + name, sp.factor(expression).is_positive)

# A constant Wald-entropy shift illustrates that the graph/bracket theorem
# does not require an unshifted constant entropy product.
h, q = sp.symbols('h q', positive=True)
shift = sp.symbols('shift', real=True)
rp, rm = q * (1 + h), q / (1 + h)
mass = (rp + rm) / 2
sx, sy = shift + sp.pi * rp**2, shift + sp.pi * rm**2
cold = shift + sp.pi * q**2
gap = sx - sy
tp = (rp - rm) / (4 * sp.pi * rp**2)
tm = (rm - rp) / (4 * sp.pi * rm**2)
ck.zero('Shifted RN: outer first law', tp * sp.diff(sx, h) - sp.diff(mass, h))
ck.zero('Shifted RN: inner first law', tm * sp.diff(sy, h) - sp.diff(mass, h))
ck.zero('Shifted RN: shifted product constant', (sx-shift)*(sy-shift) - sp.pi**2*q**4)
ck.zero('Shifted RN: outer endpoint', sp.limit(sx, h, 0, dir='+') - cold)
ck.zero('Shifted RN: inner endpoint', sp.limit(sy, h, 0, dir='+') - cold)
for name, expression in (
        ('S+ above limiting entropy', sx-cold),
        ('S- below limiting entropy', cold-sy),
        ('outer error below gap', gap-(sx-cold)),
        ('inner error below gap', gap-(cold-sy)),
        ('midpoint above limiting entropy', (sx+sy)/2-cold),
        ('midpoint error below half gap', gap/2-((sx+sy)/2-cold))):
    ck.true('Shifted RN: ' + name, sp.factor(expression).is_positive)
ck.zero('Shifted RN: two-sided midpoint error identity',
        (gap/2)**2 - ((sx+sy)/2-cold)**2 - (sx-cold)*(cold-sy))

# A harmless isolated critical point: the complement of the origin in R^2
# is connected and dense. Here inf gap=1; no extremal endpoint is claimed.
x, y, const = sp.symbols('x y const', real=True)
gap = 1 + x**2 + y**2
sx, sy = (const+gap)/2, (const-gap)/2
for coordinate in (x, y):
    ck.zero('Isolated critical point: outer law in ' + str(coordinate),
            2*sp.diff(sx, coordinate)-sp.diff(gap, coordinate))
    ck.zero('Isolated critical point: inner law in ' + str(coordinate),
            -2*sp.diff(sy, coordinate)-sp.diff(gap, coordinate))
ck.zero('Isolated critical point: gap identity', sx-sy-gap)
ck.zero('Isolated critical point: global graph', sx+sy-const)
ck.true('Isolated critical point: dg vanishes only at origin',
        sp.solve([sp.diff(gap,x),sp.diff(gap,y)], [x,y]) == {x:0,y:0})

# First laws plus gap closure do not alone imply a zero-temperature spacetime.
g = sp.symbols('g', positive=True)
sx, sy = const+g/2, const-g/2
ck.zero('Noncold limit: outer first law', 2*sp.diff(sx,g)-1)
ck.zero('Noncold limit: inner first law', -2*sp.diff(sy,g)-1)
ck.zero('Noncold limit: equal endpoint entropies', sp.limit(sx-sy,g,0,dir='+'))
ck.zero('Noncold limit: finite entropy', sp.limit(sx,g,0,dir='+')-const)
ck.true('Noncold limit: outer temperature nonzero', sp.limit(sp.Integer(2),g,0)==2)
ck.true('Noncold limit: inner temperature nonzero', sp.limit(sp.Integer(-2),g,0)==-2)

# A genuine theory with positive couplings and two DIFFERENT regular components.
# For each minimum u, phi=u and positive M>sqrt(W(u)) give an exact RN black hole.
# Its extremal entropy is pi W(u). Different products forbid a connection
# through the fixed-charge regular two-horizon solution manifold.
phi, r, mass = sp.symbols('phi r mass', real=True)
W = phi**4/4 - 4*phi**3/3 + 3*phi**2/2 + 3
ck.zero('Two minima: derivative', sp.diff(W,phi)-phi*(phi-1)*(phi-3))
ck.zero('Two minima: global positivity certificate', W-sp.Rational(3,4)
        -(phi-3)**2*((phi+sp.Rational(1,3))**2+sp.Rational(8,9))/4)
for point,value,hessian in ((0,3,3),(3,sp.Rational(3,4),6)):
    tag = 'Physical branch phi=%s: ' % point
    ck.zero(tag+'potential value', W.subs(phi,point)-value)
    ck.zero(tag+'stationarity', sp.diff(W,phi).subs(phi,point))
    ck.zero(tag+'positive Hessian', sp.diff(W,phi,2).subs(phi,point)-hessian)
    f = 1/W
    ck.zero(tag+'gauge coupling derivative', sp.diff(f,phi).subs(phi,point))
    a2 = 1-2*mass/r+value/r**2
    ck.zero(tag+'b equation', sp.diff(r,r,2))
    ck.zero(tag+'scalar equation', -sp.diff(W,phi).subs(phi,point)/(2*r**2))
    ck.zero(tag+'Maxwell charge', r**2*f.subs(phi,point)*(value/r**2)-1)
    ck.zero(tag+'Weyl equation', sp.diff(a2*r**2,r,2)-2)
    ck.zero(tag+'tt equation', sp.diff(r**2*sp.diff(a2,r),r)/2-value/r**2)
    ck.zero(tag+'angular equation', -sp.diff(a2*sp.diff(r**2,r),r)/2-value/r**2+1)
    ck.zero(tag+'Hamiltonian', -1+a2+r*sp.diff(a2,r)+value/r**2)
    ck.zero(tag+'extremal double root', a2.subs({mass:sp.sqrt(value),r:sp.sqrt(value)}))
    ck.zero(tag+'extremal derivative', sp.diff(a2,r).subs({mass:sp.sqrt(value),r:sp.sqrt(value)}))
ck.true('Two minima: gauge coupling globally positive',
        sp.solve(sp.diff(W,phi),phi)==[0,1,3] and all(W.subs(phi,p)>0 for p in [0,1,3]))

# Direct extremal field equations with scalar potential V in the paper's units.
x,V,Vp,Wp,dx,dphi = sp.symbols('x V Vp Wp dx dphi')
D = 1-2*x*V
variation = (Wp+x**2*Vp)*dphi+(2*x*V-1)*dx
ck.zero('Extremal: scalar equation forces D dx=0', variation.subs(Wp,-x**2*Vp)+D*dx)
ck.zero('Extremal: AdS2 radius equation', x/(x/D)-(1-2*x*V))
ck.zero('Extremal: implicit area gradient', (Wp+x**2*Vp)+(2*x*V-1)*(Wp+x**2*Vp)/D)
a,b,Q1,Q2 = sp.symbols('a b Q1 Q2', positive=True)
Wexp = Q1**2*sp.exp(a*phi)+Q2**2*sp.exp(-b*phi)
phistar = sp.log(b*Q2**2/(a*Q1**2))/(a+b)
ck.zero('Convex potential: critical point',
        sp.expand_power_base(sp.diff(Wexp,phi).subs(phi,phistar),force=True))
ck.true('Convex potential: Hessian positive', sp.diff(Wexp,phi,2).is_positive)

# A lower bound on W and positive outer b' imply a strictly positive gap floor
# when C < (pi W_min)^2. Numerical application is conditional on the branch.
delta,bh,bhp = sp.symbols('delta bh bhp', positive=True)
Wh = bh**2-delta*bh*bhp
ck.zero('Gap floor: outer constraint', -1+(delta/bh**2)*bh*bhp+Wh/bh**2)
ck.true('Gap floor: outer area exceeds W', sp.simplify(bh**2-Wh).is_positive)
C,Souter,Smin = sp.symbols('C Souter Smin', positive=True)
gap,floor = Souter-C/Souter,Smin-C/Smin
ck.zero('Gap floor: monotone difference factorization',
        gap-floor-(Souter-Smin)*(1+C/(Souter*Smin)))
ck.true('Gap floor: gap increases with outer entropy', sp.diff(gap,Souter).is_positive)
eta = sp.symbols('eta', positive=True)
ck.zero('Gap floor: normalized deficit', (floor/Smin).subs(C,eta*Smin**2)-(1-eta))
deficit = sp.symbols('deficit', positive=True)
ck.true('Gap floor: strict deficit gives positive floor',
        sp.simplify(floor.subs(C,(1-deficit)*Smin**2)).is_positive)

ck.finish()
