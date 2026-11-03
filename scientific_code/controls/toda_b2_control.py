"""Check the B2 closed form by exact reduced equations and a separate IVP.

The closed-form polynomial coefficients are inputs, not computed by the IVP.
This does not validate the repository's symmetric tuned-branch BVP solver for
asymmetric potentials. Only NumPy, SciPy and SymPy are required.
"""
import json
import os
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def run():
    x,t,q=sp.symbols('x t q',positive=True)
    u,v=(1+x)/2,(1-x)/2
    H1=sp.expand(u**3+3*q**2*t*v*u**2+3/q*t**2*v**2*u+q*t**3*v**3)
    H2=sp.expand(u**4+4/q**3*t*v*u**3+6/q*t**2*v**2*u**2+4*q*t**3*v**3*u+t**4*v**4/q**2)
    D=sp.diff(H1,x)/5/H1+sp.diff(H2,x)/10/H2 # b'/b
    F=-sp.Rational(2,5)*sp.diff(H1,x)/H1+sp.Rational(3,10)*sp.diff(H2,x)/H2
    R=x*x-1
    Vratio=t*(sp.Rational(3,5)*q*q*H2/H1**2+sp.Rational(2,5)*q**(-3)*H1**2/H2**2)
    dVratio=sp.Rational(12,5)*t*(q*q*H2/H1**2-q**(-3)*H1**2/H2**2)
    equations={
        'metric':sp.diff(D,x)+D*D+F*F,
        'scalar':sp.diff(R*F,x)-dVratio/2,
        'constraint':-1-R*D*D+sp.diff(R,x)*D+Vratio-R*F*F,
    }
    for name,expression in equations.items():
        numerator=sp.together(expression).as_numer_denom()[0]
        assert sp.Poly(sp.expand(numerator),x,t,q).is_zero,name
    assert sp.simplify(H1.subs(x,1)-1)==0
    assert sp.simplify(H2.subs(x,1)-1)==0
    assert sp.simplify(H1.subs(x,-1)-t**3*q)==0
    assert sp.simplify(H2.subs(x,-1)-t**4/q**2)==0
    # b_-^2/b_+^2=t^2 and phi_- - phi_+ = -log(q) = -2 epsilon.
    assert sp.Rational(2,5)*3+sp.Rational(1,5)*4==2
    assert sp.Rational(2,5)-sp.Rational(2,5)==0
    assert -sp.Rational(2,5)*3+sp.Rational(3,10)*4==0
    assert -sp.Rational(2,5)-sp.Rational(3,10)*2==-1
    # All coefficients of H1,H2 in u,v are strictly positive for q,t>0.
    # Therefore no interhorizon zero occurs. This statement says nothing
    # about positivity outside the horizons.
    fixture={t:sp.Rational(1,5),q:sp.Rational(3,2)}
    b=sp.sqrt(5)*H1**sp.Rational(1,5)*H2**sp.Rational(1,10)
    phi=sp.log(q)/2-sp.Rational(2,5)*sp.log(H1)+sp.Rational(3,10)*sp.log(H2)
    expressions=[b,b*D,phi,R*F]
    exact=sp.lambdify(x,[a.subs(fixture) for a in expressions],modules='numpy')
    def rhs(z,Y):
        b,bp,phi,j=Y
        R=z*z-1
        phip=j/R
        Vp=12/5*(np.exp(4*phi)-np.exp(-6*phi))
        return [bp,-b*phip*phip,phip,Vp/(2*b*b)]
    xs=np.linspace(.99,-.99,601)
    expected=np.array(exact(xs),dtype=float)
    errors=[]
    for rtol,atol in [(1e-10,1e-12),(1e-12,1e-14)]:
        result=solve_ivp(rhs,(xs[0],xs[-1]),expected[:,0],method='DOP853',
                         rtol=rtol,atol=atol,t_eval=xs)
        assert result.success,result.message
        maximum=np.max(np.abs(result.y-expected),axis=1)
        assert max(maximum)<(1e-6 if rtol==1e-10 else 1e-8),maximum
        errors.append({'rtol':rtol,'atol':atol,'max_absolute_errors':dict(zip(['b','bprime','phi','j'],map(float,maximum))),
                       'rhs_evaluations':result.nfev})
    return {'exact_equations':list(equations),'endpoint_phi_minus': '-epsilon',
            'endpoint_area_radius_product':'V_eff(phi_star)',
            'fixture':{'alpha1':4,'alpha2':-6,'Vstar':1,'bplus_squared':5,'exp_2epsilon':1.5},
            'ivp_interval':[.99,-.99],'ivp_comparisons':errors,
            'scope':'Exact B2 formula and independent IVP; not an asymmetric BVP pipeline benchmark.'}

if __name__=='__main__':
    result=run()
    print(json.dumps(result,indent=2))
    output=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT', Path(__file__).resolve().parents[1]/'results/data'))
    folder=output/'controls';folder.mkdir(parents=True,exist_ok=True)
    (folder/'toda_b2_control.json').write_text(json.dumps(result,indent=2)+'\n')
