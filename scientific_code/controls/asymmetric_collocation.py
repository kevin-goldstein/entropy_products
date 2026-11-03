from pathlib import Path
import sys, json, os
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import Legendre
from scipy.integrate import solve_bvp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bvp_collocation import _exp, _inverse

s=np.sqrt(7.); V0=3.; rp=1.; rm=.35; d=rp-rm
B=2.5*(d+np.sqrt(d*d+4*V0))/2; bm0=V0/B
V=lambda f:2*np.exp(s*f)+np.exp(-2*s*f)
Vp=lambda f:2*s*np.exp(s*f)-2*s*np.exp(-2*s*f)

def fields(b0,b1,f0,dh,h,N):
 b=np.zeros(N+2);f=np.zeros(N+2);b[:2]=b0,b1;f[0]=f0
 for k in range(N):
  inverse=_inverse(b,k)
  source=s*(_exp(s*f,k)-_exp(-2*s*f,k))
  a=np.convolve(source,np.convolve(inverse,inverse))[:k+1][k]
  f[k+1]=(a-k*(k+1)*f[k])/(dh*(k+1)**2)
  if k+2<=N:
   df=np.arange(1,k+2)*f[1:k+2]
   b[k+2]=-np.convolve(b[:k+1],np.convolve(df,df))[:k+1][k]/((k+2)*(k+1))
 return np.array([poly.polyval(h,b),poly.polyval(h,poly.polyder(b)),poly.polyval(h,f),poly.polyval(h,poly.polyder(f))])

def r_of(z):return rm+d*bm0*(z+1)/(B*(1-z)+bm0*(z+1))
def z_of(r):return 1-2*bm0*(rp-r)/(d*(bm0+(B-bm0)*(r-rm)/d))
def geometry(z):
 den=B*(1-z)+bm0*(z+1)
 return 2*d*B*bm0/den**2, -d*d*B*bm0*(1-z*z)/den**2

def rhs(z,y,p):
 rz,R=geometry(z);b,bp,f,j=y; fp=j/R
 return rz*np.array([bp,-b*fp*fp,fp,Vp(f)/(2*b*b)])

results=[]
for N,cutoff,tol in [(6,1e-4,1e-8),(8,3e-5,1e-9)]:
 a,b=z_of(rm+cutoff),z_of(rp-cutoff)
 z=np.linspace(a,b,501);r=r_of(z);rz,R=geometry(z)
 bseed=bm0+(B-bm0)*(r-rm)/d
 fseed=.44*Legendre.basis(2)(z)
 jseed=R*.44*Legendre.basis(2).deriv()(z)/rz
 y=np.array([bseed,np.full_like(z,(B-bm0)/d),fseed,jseed])
 def bc(ya,yb,p):
  amp,bm,bpm,fm=p
  plus=fields(B,(B*B-V(amp))/(d*B),amp,d,-cutoff,N)
  minus=fields(bm,bpm,fm,-d,cutoff,N)
  plus[3]*=(rp-cutoff-rp)*(rp-cutoff-rm)
  minus[3]*=(rm+cutoff-rp)*(rm+cutoff-rm)
  return np.r_[ya-minus,yb-plus]
 params=[.44,bm0,(bm0*bm0-V(.44))/(-d*bm0),.44]
 sol=solve_bvp(rhs,bc,z,y,p=params,tol=tol,max_nodes=40000,bc_tol=1e-11)
 assert sol.success,sol.message
 amp,bm,bpm,fm=sol.p
 assert abs(amp)>.1
 zz=np.linspace(a,b,9001);rr=r_of(zz);rz,R=geometry(zz)
 bb,bp,ff,j=sol.sol(zz);fp=j/R
 H=-1+(2*rr-rp-rm)*bp/bb-R*bp*bp/bb**2+V(ff)/bb**2-R*fp*fp
 yy=sol.sol(zz);der=sol.sol(zz,1);scaled=np.max(np.abs(der-rhs(zz,yy,sol.p))/np.maximum(1,np.abs(rhs(zz,yy,sol.p))))
 record=dict(order=N,cutoff=cutoff,tolerance=tol,phi_plus=amp,phi_minus=fm,b_minus=bm,eta=(B*bm/V0)**2,inner_constraint=-1-d*bpm/bm+V(fm)/bm**2,max_scaled_collocation_residual=scaled,max_b2H=float(np.max(np.abs(bb*bb*H))),minimum_b=float(np.min(bb)),nodes=int(np.sum(ff[:-1]*ff[1:]<0)),mesh_nodes=len(sol.x),seed='Analytic RN radius and degree-two Legendre scalar; no shooting profiles used')
 results.append(record)
 print(json.dumps(record,indent=2),flush=True)
out=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',Path(__file__).resolve().parents[1]/'results/data'))/'asymmetric'
out.mkdir(parents=True,exist_ok=True)
(out/'independent_collocation.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['eta']>1.02 and r['nodes']==2 and r['max_b2H']<3e-10 for r in results)
print('[PASS] Independent collocation confirms a two-node entropy excess')
