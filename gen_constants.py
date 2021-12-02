import numpy as np
import scipy.constants as constants

k = constants.value('Boltzmann constant in eV/K')
q = constants.e

def Vth(T):
    return k*T

def Ni(Vt,Nc,Nv,Eg):
    return (Nc*Nv*np.exp(-Eg/Vt))**0.5

def V_i(Eg,Vt,Nv,Nc):
    return 0.5*(Eg+Vt*np.log(Nc/Nv))

def Ld(eps,Vt,N):
    return (eps*Vt/(q*N))**0.5

def E_field(Vt,L):
    return Vt/L

def non_equilibrium_constants(eps,mu_n,L,ni):
    t0 = eps/(q*mu_n*ni)
    R0 = ni/t0
    J0 = R0*q*L
    return t0,J0,R0


class material:

    def __init__(self,parameters):
        T,K,Nc,Nv,Eg,mu_n,mu_p,tn,tp,affinity = parameters
        eps_0 = 0.01*constants.epsilon_0 # F/cm
        eps = K*eps_0  # F/cm
        Vt = Vth(T) # Vt at temperature T
        ni = Ni(Vt,Nc,Nv,Eg) # /cm^3 
        NcNv = (Nc*Nv)**0.5 # /cm^3
        Vi = V_i(Eg,Vt,Nv,Nc) # V
        Ld_i = Ld(eps,Vt,ni) # cm
        Ld_NcNv = Ld(eps,Vt,NcNv) # cm 
        E_field_i = E_field(Vt,Ld_i) # V/cm

        t0,J0,R0 = non_equilibrium_constants(eps,mu_n,Ld_i,ni) # s,A/cm^2,/cm^3 per second
        t0_us, to_ms = 1e6*t0,1e3*t0
    
        self.T, self.k, self.Nc, self.Nv, self.Eg, self.mu_n, self.mu_p, self.tn, self.tp,  self.affinity  = parameters
        self.Eg, self.Vi, self.Vt, self.ni, self.Ld_i, self.t0_us, self.J0, self.R0 = Eg, Vi, Vt, ni, Ld_i, t0_us, J0, R0 
