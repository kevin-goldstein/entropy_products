"""Independent raw-action, Routhian, Weyl, and RN--AdS controls.

Run with Python + SymPy. No manuscript output or solved profile is imported.
"""
from pathlib import Path
import os
import sympy as s

OUT = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"controls"
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def check(name, value):
    residual = s.factor(s.cancel(s.together(value)))
    if residual != 0:
        residual = s.simplify(residual)
    assert residual == 0, (name, residual)
    lines.append('PASS ' + name)


# Start from the sphere integral of the original Maxwell + theta action.
b, N = s.symbols('b N', positive=True)
f11, f12, f22 = s.symbols('f11 f12 f22')
t11, t12, t22 = s.symbols('theta11 theta12 theta22')
q1, q2, p1, p2, e1, e2 = s.symbols('q1 q2 p1 p2 e1 e2')
f = s.Matrix([[f11, f12], [f12, f22]])
theta = s.Matrix([[t11, t12], [t12, t22]])
q, p, e = s.Matrix([q1,q2]), s.Matrix([p1,p2]), s.Matrix([e1,e2])
dot = lambda a,c: (a.T*c)[0]
raw = b*b/(2*N)*dot(e,f*e)-N/(2*b*b)*dot(p,f*p)-dot(e,theta*p)
momentum = s.Matrix([s.diff(raw, ee) for ee in e])
estar = N/b**2*f.inv()*(q+theta*p)
sube = dict(zip(e, estar))
for i in range(2):
    check(f'raw Maxwell momentum component {i}',
          (momentum[i]-q[i]).subs(sube, simultaneous=True))
W = dot(q+theta*p, f.inv()*(q+theta*p))+dot(p,f*p)
routh = (raw-dot(q,e)).subs(sube, simultaneous=True)
check('raw electric Legendre transform equals -N W/(2 b^2)',
      routh+N*W/(2*b*b))
for zeta in [N,b,f11,f12,f22,t11,t12,t22]:
    check('partial Legendre derivative for '+str(zeta),
          s.diff(routh,zeta)-s.diff(raw,zeta).subs(sube, simultaneous=True))
check('raw Maxwell rigid Weyl weight at fixed E',
      (2*N*s.diff(raw,N)).subs(sube, simultaneous=True)+N*W/b**2)

# Derive the local normal-metric Weyl variation from the radial 2D curvature.
r, eps = s.symbols('r eps')
nf, ff, bf, sig = [s.Function(n)(r) for n in ['N','F','b','sigma']]
K, Vf, Wf = [s.Function(n)(r) for n in ['K','V','W']]


def ricci2(n,fv):
    return -s.diff(fv,r,2)-3*s.diff(fv,r)*s.diff(n,r)/n-2*fv*s.diff(n,r,2)/n


def radial_lagrangian(n,fv):
    return n/s.Integer(4)*(bf**2*ricci2(n,fv)+2*fv*s.diff(bf,r)**2+2
        -2*bf**2*fv*K-2*bf**2*Vf-2*Wf/bf**2)


weyl = s.diff(radial_lagrangian(s.exp(2*eps*sig)*nf,
                               s.exp(-2*eps*sig)*ff),eps).subs(eps,0).doit()
weyl_expected = nf*sig*(1-bf**2*Vf-Wf/bf**2)-bf**2*s.diff(nf*ff*s.diff(sig,r),r)/2
check('local Weyl variation from explicit R2(N,F)',weyl-weyl_expected)
check('rigid Weyl variation after local derivative terms vanish',
      weyl.subs({sig:1,s.diff(sig,r):0,s.diff(sig,r,2):0}).doit()
      -nf*(1-bf**2*Vf-Wf/bf**2))
# Integrating the local variation twice gives the angular Einstein equation.
local_weyl_euler = s.diff(weyl,sig)-s.diff(s.diff(weyl,s.diff(sig,r)),r)
local_weyl_euler += s.diff(s.diff(weyl,s.diff(sig,r,2)),r,2)
check('local Weyl Euler expression is angular Einstein equation',
      local_weyl_euler/nf-(1-bf**2*Vf-Wf/bf**2
      -s.diff(nf*ff*s.diff(bf**2,r),r)/(2*nf)))

# The scalar equation follows from an arbitrary target metric in one chart.
phi = s.Function('phi')(r)
gphi, Vphi, Wphi = [s.Function(n)(phi) for n in ['g','Vscalar','Wcharge']]
lscalar = -nf*bf**2*ff*gphi*s.diff(phi,r)**2/2-nf*bf**2*Vphi/2-nf*Wphi/(2*bf**2)
euler_phi = s.diff(lscalar,phi)-s.diff(s.diff(lscalar,s.diff(phi,r)),r)
target_phi = s.diff(nf*bf**2*ff*gphi*s.diff(phi,r),r)
target_phi -= nf*bf**2*ff*s.diff(gphi,phi)*s.diff(phi,r)**2/2
target_phi -= nf*bf**2*s.diff(Vphi,phi)/2+nf*s.diff(Wphi,phi)/(2*bf**2)
check('scalar Euler equation including target metric derivative',euler_phi-target_phi)

# Start independently from the full AdS2 x S2 raw action, before eliminating e.
v1, v2, V0, Ws = s.symbols('v1 v2 V W', positive=True)
fnh = (v1-v2-v1*v2*V0)/2 + v2/(2*v1)*dot(e,f*e)
fnh -= v1/(2*v2)*dot(p,f*p)+dot(e,theta*p)
enh = v1/v2*f.inv()*(q+theta*p)
en_sub = dict(zip(e,enh))
for i in range(2):
    check(f'NH raw electric charge equation {i}',
          (s.diff(fnh,e[i])-q[i]).subs(en_sub, simultaneous=True))
entropy = 2*s.pi*(dot(q,e)-fnh).subs(en_sub, simultaneous=True)
entropy_expected = s.pi*(v2-v1+v1*W/v2+v1*v2*V0)
check('NH raw action Legendre transform to entropy function',entropy-entropy_expected)
check('NH v1 derivative gives sphere radius constraint',
      s.diff(entropy,v1)/s.pi-(-1+W/v2+v2*V0))
check('NH v2 derivative gives AdS2 radius constraint',
      s.diff(entropy,v2)/s.pi-(1-v1*W/v2**2+v1*V0))
# Coupling variation tests the scalar equation without assuming a particular f.
for zeta in [f11,f12,f22,t11,t12,t22]:
    check('NH scalar coupling derivative '+str(zeta),
          s.diff(entropy,zeta)/s.pi-v1/v2*s.diff(W,zeta))
check('NH stationary entropy equals pi v2 modulo v1 equation',
      entropy-s.pi*v2-s.pi*v1*(-1+W/v2+v2*V0))
check('NH scalar-potential derivative coefficient',
      s.diff(entropy,V0)-s.pi*v1*v2)

# Verify the finite-gap real-map projections from the angular raw density.
nv,wv,vv,vp,wp,kappa,Apos,Tkin=s.symbols('n w v vp wp kappa A T',positive=True)
Xa=s.Matrix(s.symbols('X1 X2'))
upa=s.Matrix(s.symbols('up1 up2'))
angular8 = -2*nv*wv+2*vp*wp/nv+wv*vp**2/(2*nv*vv)
angular8 += -2*vv*wv/nv*Tkin-2*nv*vv*wv*V0
angular8 += 2*nv*wv/vv*dot(Xa,f*Xa)-2*vv/(nv*wv)*dot(upa,f*upa)
angular8 -= 4*dot(Xa,theta*upa)
angular=angular8/8
common=nv*s.diff(angular,nv)+wv*s.diff(angular,wv)+wp*s.diff(angular,wp)
common_expected=(nv*wv/vv*dot(Xa,f*Xa)+vv/(nv*wv)*dot(upa,f*upa)
                 -nv*wv-nv*vv*wv*V0)/2
check('angular raw density under common horizon Weyl scaling',common-common_expected)
mapping={nv:1/s.sqrt(Apos),wv:s.sqrt(Apos)/kappa,vv:b*b,V0:-V0}
mapping.update(dict(zip(Xa,-p)))
mapping.update(dict(zip(upa,estar.subs(N,1)/kappa)))
check('finite-gap metric projection under real map with V flip',
      common.subs(mapping, simultaneous=True)-(W/b**2+b**2*V0-1)/(2*kappa))
df11,df12,df22,dt11,dt12,dt22,Vprime,Gprime=s.symbols('df11 df12 df22 dt11 dt12 dt22 Vprime Gprime')
df=s.Matrix([[df11,df12],[df12,df22]])
dt=s.Matrix([[dt11,dt12],[dt12,dt22]])
dW=sum(s.diff(W,x)*dx for x,dx in zip(
    [f11,f12,f22,t11,t12,t22],[df11,df12,df22,dt11,dt12,dt22]))
ang_scalar=vv*wv/nv*Gprime-nv*vv*wv*Vprime
ang_scalar += -nv*wv/vv*dot(Xa,df*Xa)+vv/(nv*wv)*dot(upa,df*upa)
ang_scalar += 2*dot(Xa,dt*upa)
mapped_scalar=ang_scalar.subs(mapping, simultaneous=True)
source_scalar=-Apos*b*b*Gprime+dW/b**2+b*b*Vprime
check('finite-gap scalar projection including target metric and axion derivative',
      mapped_scalar+source_scalar/kappa)
# The sum of electric and magnetic horizon work always reproduces W.
Ebg=estar.subs(N,1)
dual_force=f*p/b**2+theta*Ebg
check('general matrix sum of horizon electromagnetic work',
      dot(q,Ebg)+dot(p,dual_force)-W/b**2)

# Actual nonextremal RN--AdS control, with exact horizons r-=1, r+=2.
rm, rp, ell2, mass, charge2 = map(s.Rational,['1','2','25','9/5','64/25'])
potential = -3/ell2
F = 1-2*mass/r+charge2/r**2+r**2/ell2
check('RN AdS inner horizon',F.subs(r,rm))
check('RN AdS outer horizon',F.subs(r,rp))
check('RN AdS zero metric average from direct integral',
      s.integrate(charge2/r**2+r**2*potential-1,(r,rm,rp)))
second = s.integrate(charge2/r**2-r**2*potential,(r,rm,rp))
boundary = (r**2*s.diff(F,r)).subs(r,rp)/2-(r**2*s.diff(F,r)).subs(r,rm)/2
check('RN AdS nonzero second metric average vs horizon derivatives',second-boundary)
check('RN AdS second integral is nonzero 39/25',second-s.Rational(39,25))
check('RN AdS relation to integrated scalar potential',
      boundary-(rp-rm-2*s.integrate(r**2*potential,(r,rm,rp))))

# Independently locate an extremal RN--AdS double root and its curvature.
radius = s.sqrt(3)
ext_mass, ext_charge2 = s.Rational(31,25)*s.sqrt(3),s.Rational(102,25)
Fext=1-2*ext_mass/r+ext_charge2/r**2+r**2/ell2
check('RN AdS extremal root',Fext.subs(r,radius))
check('RN AdS extremal double root',s.diff(Fext,r).subs(r,radius))
actual_v2 = radius**2
actual_v1 = 2/s.diff(Fext,r,2).subs(r,radius)
check('RN AdS v1 from actual throat curvature',actual_v1-s.Rational(75,43))
check('RN AdS extremal sphere radius equation',ext_charge2/actual_v2+actual_v2*potential-1)
check('RN AdS extremal AdS2 radius equation',1/actual_v1-ext_charge2/actual_v2**2+potential)
Etest=s.pi*(v2-v1+v1*Ws/v2+v1*v2*V0)
ev={v1:actual_v1,v2:actual_v2,Ws:ext_charge2,V0:potential}
check('RN AdS extremal entropy function stationary in v1',s.diff(Etest,v1).subs(ev))
check('RN AdS extremal entropy function stationary in v2',s.diff(Etest,v2).subs(ev))
check('RN AdS extremal entropy from raw geometry',Etest.subs(ev)-s.pi*radius**2)

# Extract the two exponential weights from W, then solve the averaged system.
a1,a2,Q,P,ph,d,IV,JV,A,B=s.symbols('a1 a2 Q P phi d IV JV A B')
Wtwo=Q**2*s.exp(a1*ph)+P**2*s.exp(a2*ph)
check('two exponential scalar weights derived by differentiating W',
      s.diff(Wtwo,ph)-a1*Q**2*s.exp(a1*ph)-a2*P**2*s.exp(a2*ph))
sol=s.solve([A+B-d+IV,a1*A+a2*B+JV],[A,B])
check('two exponential electric flux deformation',sol[A]-(-a2*(d-IV)-JV)/(a1-a2))
check('two exponential magnetic flux deformation',sol[B]-(a1*(d-IV)+JV)/(a1-a2))
# An exact asymmetric two-field RN--AdS solution with phi=0 and W'=0.
# a1=2, a2=-6, Q^2=3, P^2=1, ell^2=7, M=18/7, r-=1, r+=2.
Ftwo=1-s.Rational(36,7)/r+4/r**2+r**2/7
check('two exponential RN control inner horizon',Ftwo.subs(r,1))
check('two exponential RN control outer horizon',Ftwo.subs(r,2))
check('two exponential RN control scalar equation',s.diff(Wtwo,ph).subs({a1:2,a2:-6,Q:s.sqrt(3),P:1,ph:0}))
aval=s.integrate(3/r**2,(r,1,2))
bval=s.integrate(1/r**2,(r,1,2))
iv=s.integrate(-s.Rational(3,7)*r**2,(r,1,2))
check('two exponential exact RN AdS electric potential integral',sol[A].subs({a1:2,a2:-6,d:1,IV:iv,JV:0})-aval)
check('two exponential exact RN AdS magnetic potential integral',sol[B].subs({a1:2,a2:-6,d:1,IV:iv,JV:0})-bval)

lines.append(f'\n{len(lines)} checks passed.')
report='\n'.join(lines)+'\n'
(OUT/'check_static_virial.log').write_text(report)
print(report,end='')
