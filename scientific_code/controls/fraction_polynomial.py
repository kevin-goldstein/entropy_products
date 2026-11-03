"""Small exact rational polynomial backend; coefficients have ascending degree."""
from fractions import Fraction

class FractionPolynomial:
    def __init__(self, coefficients=0):
        if isinstance(coefficients,FractionPolynomial):
            self.coefficients=coefficients.coefficients
            return
        if isinstance(coefficients,(int,Fraction)):
            coefficients=[coefficients]
        values=list(map(Fraction,coefficients)) or [Fraction(0)]
        while len(values)>1 and not values[-1]: values.pop()
        self.coefficients=tuple(values)
    def __getitem__(self,k):
        return self.coefficients[k] if 0<=k<len(self.coefficients) else Fraction(0)
    def __iter__(self): return iter(self.coefficients)
    def __len__(self): return len(self.coefficients)
    def __bool__(self): return not self.is_zero()
    def is_zero(self): return self.coefficients==(Fraction(0),)
    def degree(self): return len(self.coefficients)-1 if not self.is_zero() else -1
    def __add__(self,other):
        other=type(self)(other)
        return type(self)([self[k]+other[k] for k in range(max(len(self),len(other)))])
    __radd__=__add__
    def __neg__(self): return type(self)([-a for a in self])
    def __sub__(self,other): return self+-type(self)(other)
    def __rsub__(self,other): return type(self)(other)+-self
    def __mul__(self,other):
        other=type(self)(other)
        result=[Fraction(0)]*(len(self)+len(other)-1)
        for i,a in enumerate(self):
            if not a: continue
            for j,b in enumerate(other):
                if b: result[i+j]+=a*b
        return type(self)(result)
    __rmul__=__mul__
    def __truediv__(self,other): return type(self)([a/Fraction(other) for a in self])
    def __pow__(self,n):
        if n<0 or not isinstance(n,int):raise ValueError('nonnegative integer power required')
        result=type(self)(1);base=self
        while n:
            if n%2:result=result*base
            base=base*base;n//=2
        return result
    def __eq__(self,other):
        try:return self.coefficients==type(self)(other).coefficients
        except (TypeError,ValueError):return False
    def __call__(self,x):
        x=Fraction(x);result=Fraction(0)
        for a in reversed(self.coefficients):result=result*x+a
        return result
    def derivative(self):return type(self)([k*self[k] for k in range(1,len(self))])
    def integral(self):return type(self)([Fraction(0)]+[self[k]/(k+1) for k in range(len(self))])
    @classmethod
    def legendre_p(cls,n):
        previous,current=cls(1),cls([0,1])
        if n==0:return previous
        for k in range(1,n):
            previous,current=current,((2*k+1)*cls([0,1])*current-k*previous)/(k+1)
        return current
    def __repr__(self):return f'FractionPolynomial({list(self.coefficients)!r})'
