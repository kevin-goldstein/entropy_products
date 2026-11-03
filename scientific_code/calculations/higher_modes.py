"""Exact polynomial checks for the symmetric higher-mode deficit branches.

Requires Python 3 and SymPy. Run from this directory with:
    python higher_modes.py > higher_modes_results.tsv

For every n=1,...,20, compute the cubic regularity coefficient and independently
reconstruct the quartic metric response. All calculations use rational numbers.
The finite sign check does not assert a sign theorem for untested modes.
"""

import sympy as s

z = s.symbols('z')

def integ(poly):
    p = s.Poly(s.expand(poly), z)
    return sum(c*s.Rational(2, k[0]+1)
               for k, c in p.terms() if k[0] % 2 == 0)

def coefficient(n, direct=False):
    L = s.Integer(n*(n+1))
    p = s.legendre(n, z)
    primitive = s.integrate(s.integrate(-s.diff(p,z)**2,z),z)
    beta = s.expand(primitive +
        (-L/2-s.diff(primitive,z).subs(z,1))*(z-1)-primitive.subs(z,1))
    assert beta.subs(z,-1) == 0
    source0 = s.expand(L**2*p**3/3-2*L*beta*p)
    L2 = s.factor(-integ(p*source0)/integ(p**2))
    eta4 = s.factor(L2/(2*n+1))
    if direct:
        source = s.Poly(s.expand(source0+L2*p), z)
        qcoeff = {}
        for k in range(source.degree(), -1, -1):
            rhs = source.nth(k)+(k+2)*(k+1)*qcoeff.get(k+2,0)
            if k == n:
                assert rhs == 0
                qcoeff[k] = 0
            else:
                qcoeff[k] = s.factor(rhs/(k*(k+1)-L))
        q = sum(c*z**k for k,c in qcoeff.items())
        q = s.expand(q-q.subs(z,1)*p)
        assert q.subs(z,-1) == 0
        assert s.expand(s.diff((z*z-1)*s.diff(q,z),z)-L*q-source.as_expr()) == 0
        primitive = s.integrate(s.integrate(
            -beta*s.diff(p,z)**2-2*s.diff(p,z)*s.diff(q,z),z),z)
        gamma = s.expand(primitive +
            (-L2/2-L**2/12-s.diff(primitive,z).subs(z,1))*(z-1)-primitive.subs(z,1))
        assert 2*gamma.subs(z,-1) == eta4
    return L2, eta4

if __name__ == '__main__':
    print('n\tLambda_2\tc_n')
    for n in range(1,21):
        L2, eta4 = coefficient(n, direct=True)
        assert eta4 == 0 if n <= 2 else eta4 < 0
        print(f'{n}\t{L2}\t{eta4}')
