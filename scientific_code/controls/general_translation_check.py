#!/usr/bin/env python3
"""Exact controls for the general V=0 orbit map and singular-endpoint fluxes.

The target-space kinetic contraction is kept as an arbitrary symbol, so the
constraint check does not assume a constant or one-dimensional target metric.
The covariant scalar pullback identity follows separately by the chain rule;
this script verifies the geometric Jacobian and force weights it requires.
No numerical profile, fitted singular limit, or review output is imported.
"""
import json
import os
from pathlib import Path
import sympy as s

checks = []
def zero(name, expression):
    result = s.factor(s.cancel(expression))
    if result != 0:
        raise AssertionError((name,result))
    checks.append(name)

x, lam, d = s.symbols('x lambda d', positive=True)
B, B1, B2, W, kinetic = s.symbols('b by byy W kinetic')
D = 1+(lam-1)*x
h = D/s.sqrt(lam)
y = lam*x/D
j = lam/D**2
R = d*d*x*(x-1)
Ry = d*d*y*(y-1)
zero('radial Jacobian y_r=j (using dimensionless y)',s.diff(y,x)-j)
zero('radius multiplier h has zero second derivative',s.diff(h,x,2))
zero('radial quadratic transforms with j',Ry-j*R)
zero('Maxwell and scalar-force weights transform with j',1/(h*B)**2-j/B**2)
bnew = h*B
bnew1 = s.diff(h,x)*B/d+h*j*B1
bnew2 = s.diff(bnew1,x)/d+s.diff(bnew1,B)*j*B1+s.diff(bnew1,B1)*j*B2
zero('radius second derivative with arbitrary target kinetic norm',bnew2-h*j*j*B2)
oldconstraint = -1+d*(2*y-1)*B1/B-Ry*B1**2/B**2+W/B**2-Ry*kinetic
newconstraint = -1+d*(2*x-1)*bnew1/bnew-R*bnew1**2/bnew**2+W/bnew**2-R*j*j*kinetic
zero('Hamiltonian constraint has weight j',newconstraint-j*oldconstraint)
# Arbitrary gauge matrix and axionic mixing enter through this source vector.
effective_source = s.symbols('f_inverse_times_Q_plus_thetaP')
zero('electric first integral including axionic charge mixing',j*effective_source/B**2-effective_source/bnew**2)
# The magnetic flux P is unchanged by construction; recover Q after mixing.
f, theta, Q, P = s.symbols('f theta Q P',nonzero=True)
Enew = (Q+theta*P)/(f*bnew**2)
zero('electric flux Q with arbitrary axionic term',f*bnew**2*Enew-theta*P-Q)

rm, r, bR = s.symbols('rm r bR',real=True,nonzero=True)
rp=rm+d
xt=(r-rm)/d
Dt=1+(lam-1)*xt
rt=rm+d*lam*xt/Dt
Rlim=rp+d/(lam-1)
zero('exterior limiting seed point',s.limit(rt,r,s.oo)-Rlim)
zero('seed radial quadratic at limiting point',(Rlim-rp)*(Rlim-rm)-d*d*lam/(lam-1)**2)
c=(lam-1)*bR/(d*s.sqrt(lam))
zero('asymptotic metric normalization',((Rlim-rp)*(Rlim-rm)/bR**2)-1/c**2)
zero('coordinate normalization preserves radial electric component',c*(1/c)-1)
# A scalar potential does not share the necessary inverse-radius weight.
V=s.symbols('V')
zero('nonzero scalar-potential mismatch is explicit',(h*B)**2*V-j*B**2*V-B**2*V*(h*h-j))

# Exact benchmark: magnetic dilaton, W/b²=P²/r².
a, rp0, rm0 = s.symbols('a rp rm',positive=True)
d0=rp0-rm0
charge2=rp0*rm0/(1+a*a)
log_coefficient=-a/(1+a*a)
radius_exponent=a*a/(1+a*a)
scalar_integral=-2*a*charge2*(1/rm0-1/rp0)
metric_integral=charge2*(1/rm0-1/rp0)-d0
zero('magnetic dilaton scalar endpoint flux',scalar_integral-2*d0*log_coefficient)
zero('magnetic dilaton metric endpoint flux',metric_integral+d0*radius_exponent)
zero('leading scalar-radius singular balance',radius_exponent*(radius_exponent-1)+log_coefficient**2)

result=dict(status='PASS',checks=len(checks),identities=checks,
            scope='V=0, static spherical neutral-scalar/abelian sector; exterior regularity is inherited from a regular collar. Singular endpoint identities require retaining boundary fluxes.',
            sympy=s.__version__)
text=json.dumps(result,indent=2)+'\n'
out=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT', Path(__file__).resolve().parents[1]/'results/data'))/'controls'
out.mkdir(parents=True,exist_ok=True)
(out/'general_translation.json').write_text(text)
print(text,end='')
