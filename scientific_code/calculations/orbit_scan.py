"""Independent fixed-alpha translation-orbit/exterior reconstruction.

Uses only SciPy and NumPy. Launch with a Python containing these packages.
All quantities use G=1; scalar charge is phi=phi_inf+Sigma/R+O(R^-2).
The exterior equations are implemented separately from the interhorizon solvers.
The orbit seed is freshly computed by half-interval shooting; agreement with
stored reference values is a regression check, not an independent derivation.
"""
import json, csv, math
from pathlib import Path
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq

ALPHA=None
RP,RM=1.,.35
D=RP-RM
BDE=(D+np.sqrt(D*D+8))/2
BREF=2.5*BDE
PHIP=.2
BMINUS_REF=PRODUCT_RADIUS=ETA=H=None
SEED_SOURCE=None

def configure_seed(alpha,bminus,source):
    """Set freshly computed interior data without changing the exterior method."""
    global ALPHA,BMINUS_REF,PRODUCT_RADIUS,ETA,H,SEED_SOURCE
    if not np.all(np.isfinite([alpha,bminus])) or alpha<=0 or bminus<=0:
        raise ValueError('Positive finite interior seed data required')
    ALPHA=float(alpha);BMINUS_REF=float(bminus)
    PRODUCT_RADIUS=BREF*BMINUS_REF
    ETA=PRODUCT_RADIUS**2/4
    H=1+ALPHA**2/4
    SEED_SOURCE=source

def ensure_seed():
    if ALPHA is None:
        from half_interval_scan import tuned
        row=tuned(PHIP,cutoff=1e-7,order=5,rtol=3e-14)
        configure_seed(row['alpha'],row['bminus'],'fresh half-interval shooting')

def rhs_tau(t,y):
    u,v,p,w=y
    wp=np.exp(2*u+ALPHA*p)
    wm=np.exp(2*u-ALPHA*p)
    return [v,wp+wm,w,ALPHA*(wp-wm)/2]

def get_reference(rtol=2e-13,e=1e-6):
    ensure_seed()
    # Local outer Taylor data through the terms needed for an O(e^3) start.
    a=ALPHA;b=BREF;p=PHIP;dh=D
    W=2*np.cosh(a*p);Wp=2*a*np.sinh(a*p);Wpp=a*a*W
    b1=(b-W/b)/dh
    p1=Wp/(2*b*b*dh)
    b2=-b*p1*p1/2
    p2=(Wpp*p1/(2*b*b)-Wp*b1/(b**3)-2*p1)/(4*dh)
    b3=-(b1*p1*p1+4*b*p1*p2)/6
    # scalar recursion from manuscript A.3, expanded directly to second order.
    Wppp=a*a*Wp
    src2= Wpp*p2/(2*b*b)+Wppp*p1*p1/(4*b*b)-Wpp*p1*b1/(b**3)+Wp*(3*b1*b1/b**2-2*b2/b)/(2*b*b)
    p3=(src2-6*p2)/(9*dh)
    bv=b+b1*e+b2*e*e+b3*e**3
    vp=b1+2*b2*e+3*b3*e*e
    ph=p+p1*e+p2*e*e+p3*e**3
    phip=p1+2*p2*e+3*p3*e*e
    r=RP+e;P=(r-RP)*(r-RM);Pp=2*r-RP-RM
    tau=np.log((r-RP)/(r-RM))/D
    init=[.5*np.log(P)-np.log(bv),Pp/2-P*vp/bv,ph,P*phip]
    sol=solve_ivp(rhs_tau,[tau,0],init,method='DOP853',rtol=rtol,atol=rtol/100,dense_output=True)
    assert sol.success
    return sol

def compact_rhs(t,q):
    y,z,k=q
    # y=e^{-(U+alpha phi/2)}; z=phi-alpha U/2; k=z_tau.
    # The endpoint y=0 is regular in these variables.
    yp=max(0.,y)
    wm=yp**(2*(ALPHA**2/4-1)/H)*np.exp(-2*ALPHA*z/H)
    rad=H+yp*yp*(H*wm+H*D*D/4-k*k)
    assert rad>0
    return [-np.sqrt(rad),k,-ALPHA*wm]

def endpoint_event(t,q):return q[0]
endpoint_event.terminal=True
endpoint_event.direction=-1

def get_compact(ref,rtol=2e-13):
    u,v,p,w=ref.y[:,-1]
    yy=np.exp(-u-ALPHA*p/2);z=p-ALPHA*u/2;k=w-ALPHA*v/2
    sol=solve_ivp(compact_rhs,[0,1],[yy,z,k],method='DOP853',rtol=rtol,atol=rtol/100,
                 dense_output=True,events=endpoint_event)
    assert sol.success and len(sol.t_events[0])==1
    return sol

def make_data(ref,comp,B):
    if not np.isfinite(B) or B <= 0:
        raise ValueError('Positive finite outer radius required')
    tau=-2*np.log(B/BREF)/D
    if tau < ref.t[0] or tau >= comp.t_events[0][0]:
        raise ValueError('Orbit member lies outside computed regular trajectory')
    if tau<=0:
        u,v,p,w=ref.sol(tau)
        c=np.exp(-u)
    else:
        y,z,k=comp.sol(tau)
        ydot=compact_rhs(tau,[y,z,k])[0]
        u=(-np.log(y)-ALPHA*z/2)/H
        p=(z-ALPHA*np.log(y)/2)/H
        v=(-ydot/y-ALPHA*k/2)/H
        w=(k-ALPHA*ydot/(2*y))/H
        c=np.exp(-u)
    bm=PRODUCT_RADIUS/B
    sp=np.pi*B*B;sm=np.pi*bm*bm
    tp=c*D/(4*np.pi*B*B);tm=-c*D/(4*np.pi*bm*bm)
    return dict(B=B,B_over_BDE=B/BDE,tau=tau,c=c,M=c*v,Tplus=tp,Tminus=tm,
                phi_inf=p,Sigma=-c*w,Splus=sp,Sminus=sm,
                gap_over_Sstar=(sp-sm)/(2*np.pi),bminus=bm)

def main():
    ref=get_reference();comp=get_compact(ref)
    ts=comp.t_events[0][0];bc=BREF*np.exp(-D*ts/2)
    zc,kc=comp.y_events[0][0,1:]
    hconstraint=(ref.y[1]**2+ref.y[3]**2
       -np.exp(2*ref.y[0]+ALPHA*ref.y[2])-np.exp(2*ref.y[0]-ALPHA*ref.y[2])-D*D/4)
    meta=dict(alpha=ALPHA,eta=ETA,BDE=BDE,Bref=BREF,product_radius=PRODUCT_RADIUS,
              seed_source=SEED_SOURCE,bminus_reference=BMINUS_REF,
              tau_endpoint=ts,Bcrit=bc,Bcrit_over_BDE=bc/BDE,
              Spluscrit=np.pi*bc*bc,Sminuscrit=np.pi*(PRODUCT_RADIUS/bc)**2,
              gapcrit_over_Sstar=(bc*bc-(PRODUCT_RADIUS/bc)**2)/2,
              z_endpoint=zc,k_endpoint=kc,H=H,power_temperature=1/H,
              power_mass=1/H-1,max_H_error=float(max(abs(hconstraint))))
    opt=minimize_scalar(lambda B:-make_data(ref,comp,B)['Tplus'],bounds=(bc+1e-9,10),method='bounded',options={'xatol':1e-12})
    meta['temperature_maximum']=make_data(ref,comp,opt.x)
    optm=minimize_scalar(lambda B:make_data(ref,comp,B)['M'],bounds=(bc+1e-9,10),method='bounded',options={'xatol':1e-12})
    meta['mass_minimum']=make_data(ref,comp,optm.x)
    Bs=[bc*(1+10**(-k)) for k in [8,6,4,3,2]]
    Bs+=list(np.array([2.5,3,4])*BDE)+[5,7.5,10,20,50,100,1000]
    rows=[make_data(ref,comp,B) for B in sorted(set(Bs))]
    out=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',Path(__file__).resolve().parents[1]/'results/data'))/'orbit'
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'orbit_data.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'orbit_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    convergence=[]
    for e,rtol in [(1e-5,2e-11),(1e-6,2e-12),(1e-7,2e-13),(1e-6,2e-13)]:
        rr=get_reference(rtol,e);cc=get_compact(rr,rtol)
        tsc=cc.t_events[0][0];bcr=BREF*np.exp(-D*tsc/2)
        datum=make_data(rr,cc,BREF)
        convergence.append(dict(cutoff=e,rtol=rtol,Bcrit=bcr,taucrit=tsc,
                                Mref=datum['M'],phi_inf_ref=datum['phi_inf']))
    (out/'orbit_convergence.json').write_text(json.dumps(convergence,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    for row in rows:print(json.dumps(row))

if __name__=='__main__': main()
