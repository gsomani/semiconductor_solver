import numpy as np
import bc
from tdma import *
from scipy.linalg import solve_banded

def update_Vnp_gummel(X,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu_p,gen,iterations=10):
    for i in range(iterations):
        Vc = np.copy(V)
        R,R_ = n - p - Nd, n + p
        V[1:-1] = solve_tdma(lap[0],lap[1]-R_[1:-1],np.copy(lap[2]),(R - V*R_)[1:-1] - d)   
        dV = V - Vc 
        n = np.exp(V-fn)
        p = np.exp(-V+fp)

        V_ = (V[1:]+V[:-1])/2

        beta = tp*(n+1)+tn*(p+1)
        pn = np.exp(fp-fn)
        f = pn/beta + (1-pn)*tp*n/(beta*beta)

        an = -np.exp(V_[:-1]-fn[:-2])/dx[:-1]
        cn = -np.exp(V_[1:]-fn[2:])/dx[1:]
        bn0 = np.exp(V_[:-1]-fn[1:-1])/dx[:-1]   
        bn1 = np.exp(V_[1:]-fn[1:-1])/dx[1:]
        dn = ((pn-1)/beta - gen)[1:-1]*X + an + bn0 + bn1 + cn
        bn =  bn0 + bn1 + f[1:-1]*X
        an[0] = cn[-1] = 0

        dfn = solve_tdma(an,bn,cn,dn)
        err_n = abs(dfn/np.maximum(1,abs(fn[1:-1])))
        fn[1:-1] += dfn
        n = np.exp(V-fn)

        beta = tp*(n+1)+tn*(p+1)
        pn = np.exp(fp-fn)
        f = pn/beta + (1-pn)*tn*p/(beta*beta)

        ap = mu_p*np.exp(-V_[:-1]+fp[:-2])/dx[:-1]
        cp = mu_p*np.exp(-V_[1:]+fp[2:])/dx[1:]
        bp0 = -mu_p*np.exp(-V_[:-1]+fp[1:-1])/dx[:-1]
        bp1 = -mu_p*np.exp(-V_[1:]+fp[1:-1])/dx[1:]
        dp = ((pn-1)/beta - gen)[1:-1]*X  - (ap + bp0 + bp1 + cp)
        bp = bp0 + bp1 - f[1:-1]*X
        ap[0]=cp[-1]=0
    
        dfp = solve_tdma(ap,bp,cp,dp)
        err_p = abs(dfp/np.maximum(1,abs(fp[1:-1])))
        fp[1:-1] += dfp
        p = np.exp(-V+fp)

        err_V = abs(dV/np.maximum(1,abs(Vc)))
   
    return V,n,p,fn,fp,max(err_V),max(err_n),max(err_p)

def update_Vnp_newton(X,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu_p,gen):
    V_ = (V[1:]+V[:-1])/2

    R,R_ = (n - p - Nd)[1:-1], (n + p)[1:-1]
    beta = tp*(n+1)+tn*(p+1)
    pn = np.exp(fp-fn)
    f_dV = (pn-1)*(tn*p-tp*n)/(beta*beta)
    f_dfp = pn/beta + (1-pn)*tn*p/(beta*beta)
    f_dfn = -(f_dV+f_dfp)

    an = -np.exp(V_[:-1]-fn[:-2])/dx[:-1]
    cn = -np.exp(V_[1:]-fn[2:])/dx[1:]
    bn0 = np.exp(V_[:-1]-fn[1:-1])/dx[:-1]
    bn1 = np.exp(V_[1:]-fn[1:-1])/dx[1:]
    bn = bn0 + bn1
    dn = ((pn-1)/beta - gen)[1:-1]*X + an + bn + cn
    bn = bn - f_dfn[1:-1]*X

    ap = mu_p*np.exp(-V_[:-1]+fp[:-2])/dx[:-1]
    cp = mu_p*np.exp(-V_[1:]+fp[2:])/dx[1:]
    bp0 = -mu_p*np.exp(-V_[:-1]+fp[1:-1])/dx[:-1]
    bp1 = -mu_p*np.exp(-V_[1:]+fp[1:-1])/dx[1:] 
    bp = bp0 + bp1    
    dp = ((pn-1)/beta - gen)[1:-1]*X  - (ap + bp + cp)
    bp = bp - f_dfp[1:-1]*X

    N = len(X)
    ab = np.zeros([9,3*N])
    b = np.zeros([3*N])

    # Poisson Equation
    ab[1,4::3] = lap[2,:-1]
    ab[4,1::3] = lap[1]-R_
    ab[7,1:-3:3] = lap[0,1:]
    ab[5,::3] = n[1:-1] 
    ab[3,2::3] = p[1:-1]

    # Electron Continuity Equation
    ab[0,4::3] = -(cn + bn1)[:-1]/2
    ab[1,3::3] = cn[:-1]
    ab[2,2::3] = -f_dfp[1:-1]*X
    ab[4,::3] = bn
    ab[6,1:-3:3] = -(an + bn0)[1:]/2
    ab[7,:-3:3] = an[1:]
    ab[3,1::3] = ab[0,1::3] + ab[6,1::3] - f_dV[1:-1]*X 
  
    # Hole continuity Equation 
    ab[1,5::3] = cp[:-1]
    ab[2,4::3] = -(cp + bp1)[:-1]/2
    ab[4,2::3] = bp
    ab[6,::3] = -f_dfn[1:-1]*X
    ab[7,2:-3:3] = ap[1:]
    ab[8,1:-3:3] = -(ap + bp0)[1:]/2
    ab[5,1::3] = ab[2,1::3] + ab[8,1::3] - f_dV[1:-1]*X

    b[::3] = dn
    b[1::3] = R - (lap[0]*V[:-2] + lap[1]*V[1:-1] + lap[2]*V[2:]) - d
    b[2::3] = dp

    nVp = solve_banded((4,4),ab,b)
    err_n = abs(nVp[::3]/np.maximum(1,abs(fn[1:-1])))
    err_p = abs(nVp[2::3]/np.maximum(1,abs(fp[1:-1])))
    err_V = abs(nVp[1::3]/np.maximum(1,abs(V[1:-1])))

    fn[1:-1] += nVp[::3]
    V[1:-1] += nVp[1::3]
    fp[1:-1] += nVp[2::3]

    n = np.exp(V-fn)
    p = np.exp(-V+fp)

    return V,n,p,fn,fp,max(err_V),max(err_n),max(err_p)

def update_Vnp_ac(X,dx,V,n,p,fn,fp,tp,tn,lap,d,mu_p,w,gen):
    V_ = (V[1:]+V[:-1])/2

    R_ = (n + p)[1:-1]
    beta = tp*(n+1)+tn*(p+1)
    pn = np.exp(fp-fn)
    f_dV = (pn-1)*(tn*p-tp*n)/(beta*beta)
    f_dfp = pn/beta + (1-pn)*tn*p/(beta*beta)
    f_dfn = -(f_dV+f_dfp)

    an = -np.exp(V_[:-1]-fn[:-2])/dx[:-1]
    cn = -np.exp(V_[1:]-fn[2:])/dx[1:]
    bn0 = np.exp(V_[:-1]-fn[1:-1])/dx[:-1]
    bn1 = np.exp(V_[1:]-fn[1:-1])/dx[1:]
    bn = bn0 + bn1
    bn = bn - f_dfn[1:-1]*X

    ap = mu_p*np.exp(-V_[:-1]+fp[:-2])/dx[:-1]
    cp = mu_p*np.exp(-V_[1:]+fp[2:])/dx[1:]
    bp0 = -mu_p*np.exp(-V_[:-1]+fp[1:-1])/dx[:-1]
    bp1 = -mu_p*np.exp(-V_[1:]+fp[1:-1])/dx[1:]
    bp = bp0 + bp1
    bp = bp - f_dfp[1:-1]*X

    N = len(X)
    ab = np.zeros([9,3*N],dtype=complex)
    b = np.zeros([3*N],dtype=complex)

    dn_dt = w*n[1:-1]*1j
    dp_dt = w*p[1:-1]*1j

    # Poisson Equation
    ab[1,4::3] = lap[2,:-1]
    ab[4,1::3] = lap[1] - R_
    ab[7,1:-3:3] = lap[0,1:]
    ab[5,::3] = n[1:-1] 
    ab[3,2::3] = p[1:-1]

    # Electron Continuity Equation
    ab[0,4::3] = -(cn + bn1)[:-1]/2
    ab[1,3::3] = cn[:-1]
    ab[2,2::3] = -f_dfp[1:-1]*X
    ab[4,::3] = bn + dn_dt
    ab[6,1:-3:3] = -(an + bn0)[1:]/2
    ab[7,:-3:3] = an[1:]
    ab[3,1::3] = ab[0,1::3] + ab[6,1::3] - f_dV[1:-1]*X - dn_dt
  
    # Hole continuity Equation 
    ab[1,5::3] = cp[:-1]
    ab[2,4::3] = -(cp + bp1)[:-1]/2
    ab[4,2::3] = bp - dp_dt
    ab[6,::3] = -f_dfn[1:-1]*X
    ab[7,2:-3:3] = ap[1:]
    ab[8,1:-3:3] = -(ap + bp0)[1:]/2
    ab[5,1::3] = ab[2,1::3] + ab[8,1::3] - f_dV[1:-1]*X + dp_dt
    
    b[::3] = -d
    b[1::3] = 0
    b[2::3] = 0

    nVp = solve_banded((4,4),ab,b)

    dfn = np.zeros(N+2,dtype=complex)
    dfp = np.zeros(N+2,dtype=complex)
    dV = np.zeros(N+2,dtype=complex)
    
    dV[0] = dfn[0] = dfp[0] = 1

    
    dfn[1:-1] = nVp[::3]
    dV[1:-1] = np.array([0.8,0.6,0.4,0.2])
    dfp[1:-1] = nVp[2::3]

    return dV,dfn,dfp
