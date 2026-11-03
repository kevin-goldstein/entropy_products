"""Exact rational-function controls for the rotating functional.

No optional packages are required.  The NHEK test sets J=1; homogeneity
restores arbitrary positive J.  The tests derive the three metric residuals
from the displayed Euler expressions after eliminating square roots using
n*w=2, w**2=4*(1-z**2)/(1+z**2), and w'/w=-2*z/(1-z**4).
"""
from fractions import Fraction as Q


def trim(p):
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def add(a, b):
    p = [Q(0)] * max(len(a), len(b))
    for i, x in enumerate(a):
        p[i] += x
    for i, x in enumerate(b):
        p[i] += x
    return trim(p)


def mul(a, b):
    p = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            p[i + j] += x * y
    return trim(p)


def deriv(a):
    return trim([Q(i) * a[i] for i in range(1, len(a))] or [Q(0)])


class R:
    def __init__(self, a=0, b=1):
        self.a = [Q(x) for x in a] if isinstance(a, (tuple, list)) else [Q(a)]
        self.b = [Q(x) for x in b] if isinstance(b, (tuple, list)) else [Q(b)]

    @staticmethod
    def convert(other):
        return other if isinstance(other, R) else R(other)

    def __add__(self, other):
        other = self.convert(other)
        return R(add(mul(self.a, other.b), mul(other.a, self.b)), mul(self.b, other.b))

    __radd__ = __add__

    def __neg__(self):
        return R([-x for x in self.a], self.b)

    def __sub__(self, other):
        return self + -self.convert(other)

    def __rsub__(self, other):
        return self.convert(other) - self

    def __mul__(self, other):
        other = self.convert(other)
        return R(mul(self.a, other.a), mul(self.b, other.b))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.convert(other)
        return R(mul(self.a, other.b), mul(self.b, other.a))

    def __rtruediv__(self, other):
        return self.convert(other) / self

    def d(self):
        return R(add(mul(deriv(self.a), self.b),
                     [-x for x in mul(self.a, deriv(self.b))]), mul(self.b, self.b))

    def value(self, z):
        evaluate = lambda p: sum(x * Q(z) ** i for i, x in enumerate(p))
        return evaluate(self.a) / evaluate(self.b)

    def zero(self, name):
        assert all(x == 0 for x in self.a), name
        print('PASS:', name)


z = R([0, 1])
t, s = 1 + z*z, 1 - z*z
w2 = 4*s/t
a = -2*z/(s*t)                 # w'/w
vp = 2*z
wp_over_n = -4*z/(t*t)
wv_over_nv = 4*z*s/(t*t)

# 8 E_v is already rational.  Multiplying 8 E_w by w, or dividing
# 8 E_n by w, removes the remaining square roots without changing zeros.
ev8 = -(w2/2)*vp*vp/(2*t*t) - 2*w2/(2*t*t) - (2*wp_over_n + wv_over_nv).d()
wew8 = -4 + w2*z*z/t + 3*w2/t - 2*w2*(1 + z*a)
en8_over_w = -2 - vp*a*w2/2 - vp*vp*w2/(8*t) + w2/(2*t)
ev8.zero('NHEK warp metric Euler equation')
wew8.zero('NHEK circle metric Euler equation')
en8_over_w.zero('NHEK angular lapse constraint')

ell = (-4 + 2*vp*wp_over_n + (w2/2)*vp*vp/(2*t) + w2/t)/8
bulk_primitive = -(z*z*z)/t
(ell - bulk_primitive.d()).zero('NHEK raw bulk integrand has exact primitive')
bulk_f = bulk_primitive.value(1) - bulk_primitive.value(-1)
bg = Q(1)
assert bulk_f == -1 and bulk_f + bg == 0
print('PASS: NHEK bulk f=-J, axis B_g=J, full f=0')

j_density = s/(t*t)
j_primitive = z/t
(j_density - j_primitive.d()).zero('NHEK rotational charge density has exact primitive')
assert j_primitive.value(1) - j_primitive.value(-1) == 1
print('PASS: df/dk=J and entropy/(2*pi)=J')

# Spherical attractor signs with arbitrary negative cosmological potential.
v2, potential = Q(1), Q(-1, 3)
charge_potential = v2 * (1-v2*potential)
v1 = v2 / (1-2*v2*potential)
assert charge_potential/v2 + v2*potential == 1
assert 1/v1 == charge_potential/(v2*v2)-potential
entropy_over_pi = v2-v1+v1*v2*potential+v1*charge_potential/v2
assert entropy_over_pi == v2
print('PASS: spherical AdS potential signs and on-shell entropy')
