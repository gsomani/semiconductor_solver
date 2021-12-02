import numpy as np
import eq
	
def eq_dirichlet(phi,Eg,Vi,Nd,Nc,Nv,ni,contact,stat):
    if(contact=='o'):
        if(stat == 'boltzmann'):
            V = phi + np.arcsinh(Nd/(2*ni))
        else:
            V = phi + eq.eq_pot_fermi(Eg,Vi,Nd,Nc,Nv,ni)
    else:
        V = Vi - phi
    return V
	
def eq_bc(contact,phi,Eg,Vi,Vt,Nd,Nc,Nv,ni,stat):
    if(contact=='n'):
        bc = ['n',phi]
    else:
        bc = ['d',eq_dirichlet(phi/Vt,Eg/Vt,Vi/Vt,Nd,Nc,Nv,ni,contact,stat)*Vt]
    return bc

def set_eq_bc_1d(contact,Eg,Vi,Vt,Nd,Nc,Nv,ni,stat='fermi'):
    N = len(Nd)
    return np.array([eq_bc(contact[0][0],contact[0][1],Eg,Vi,Vt,Nd[0],Nc,Nv,ni,stat),eq_bc(contact[1][0],contact[1][1],Eg,Vi,Vt,Nd[N-1],Nc,Nv,ni,stat)],dtype=list)

def set_bc_2d(contact_top,contact_bot,Eg,Vi,Vt,Nd,Nc,Nv,ni,stat='fermi'):
    M,N = Nd.shape
    bc_top = np.empty(N,dtype=list)
    bc_bot = np.empty(N,dtype=list)
    for i in range(N):
        bc_top[i] = eq_bc(contact_top[i][0],contact_top[i][1],Eg,Vi,Vt,Nd[0][i],Nc,Nv,ni,stat)
        bc_bot[i] = eq_bc(contact_bot[i][0],contact_bot[i][1],Eg,Vi,Vt,Nd[M-1][i],Nc,Nv,ni,stat)
    return bc_top,bc_bot
