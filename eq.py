import numpy as np
from scipy.special import roots_genlaguerre
from scipy.interpolate import CubicSpline
import mesh

c = [2**-1.5 , 3/16 - 3**-1.5]
precision = 53
d = precision*np.log(2)

def quad(N,max_E): # Gauss-Laguerre quadrature nodes and weights for numerical integration of Fermi-Dirac integral 
    q = np.empty([2,N])
    w = np.zeros([2,N])
    n = [-0.5,0.5]
    a = np.empty(2,dtype=int)
    g = np.empty(2)
    for i in range(2):
        r,w[i] = roots_genlaguerre(N,n[i])
        for j in range(N):
        	cur = r[j]
        	q[i][j] = np.exp(-cur)
        	if(cur > max_E + d):
        		w[i][j] = np.sum(w[i,j:])
        		break
        a[i] = j+1
        g[i] = np.sum(w[i])
    return q[0,:a[0]],w[0,:a[0]]/g[0],q[1,:a[1]],w[1,:a[1]]/g[1]

q0,w0,q1,w1 = quad(160,16) # Gauss-Laguerre quadrature implemented with polynomial of degree 160

def exact_F(E): # Exact evaluation of FD integral (j=0.5) using Gauss-Laguerre quadrature nodes and weights
    if(E<=-d):
        return np.exp(E)
    return np.sum(w1/(q1+np.exp(-E)))
	
def exact_F_(E,r=len(w0)): # Exact evaluation of FD integral (j=-0.5) using Gauss-Laguerre quadrature nodes and weights
    if(E<=-d):
        return np.exp(E)
    return np.sum(w0/(q0+np.exp(-E)))

loc=[-37,16]
spac=[2**-6,2**-10]
x,node = mesh.gen_mesh_1d(loc,spac)
N = len(x) # Total number of nodes

log_F=np.empty(N,dtype=np.longdouble)
log_F_=np.empty(N,dtype=np.longdouble)

x = np.array(x,dtype=np.longdouble)

# logarithm of numerically integrated FD integrals
for i in range(N):
    log_F[i] = np.log(exact_F(x[i]))
    log_F_[i] = np.log(exact_F_(x[i]))

spline_F = CubicSpline(x,log_F)
spline_F_ = CubicSpline(x,log_F_)

def F(E):
    s = np.array(spline_F(E),dtype=np.longdouble)
    return np.exp(s)
	
def F_(E):
    s = np.array(spline_F_(E),dtype=np.longdouble)
    return np.exp(s)

def inv_FD(f): # inverse FD integral using newton's method and Joyce-Dixon as initial approximation
    x = np.log(f) + c[0]*f +  c[1]*f*f
    dx = eps = 2**-45
    while(abs(dx) >= eps):
        r = f - F(x)
        dx = r/F_(x)
        x += dx
    return x

def eq_electrons(V,Vi,Eg,Nc,Nv,ni,stat='fermi'): # Equilibrium population of electrons
    if(stat=='boltzmann'):
        return ni*np.exp(V)
    return Nc*F(V-Vi)

def eq_holes(V,Vi,Eg,Nc,Nv,ni,stat='fermi'): # Equilibrium population of electrons
    if(stat=='boltzmann'):
        return ni*np.exp(-V)
    return Nv*F(-V+Vi-Eg)

def eq_pot_fermi(Eg,Vi,N,Nc,Nv,ni): # Initial guess for equilibrium potential with FD statistics
    N,Nc,Nv = N/ni,Nc/ni,Nv/ni
    if(N>0.75*Nc):
        V_ = inv_FD(N/Nc)
    elif(N<-0.75*Nv):
        V_ = -inv_FD(-N/Nv) - Eg
    else:
        V_ = np.arcsinh(N/2) - Vi
    dV = eps = 2**-45
    while(abs(dV) >= eps):
        f,df = Nc*F(V_) - Nv*F(-V_-Eg) - N, Nc*F_(V_) + Nv*F_(-V_-Eg)
        dV = f/df
        V_ -= dV
    return V_ + Vi
