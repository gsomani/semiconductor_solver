import numpy as np
import eq
from tdma import solve_tdma
import bc

def init(Eg,Vi,Vt,Nd,Nc,Nv,ni,stat='boltzmann'):
    if(stat == 'boltzmann'):
        V = np.arcsinh(Nd/(2*ni))
    else:
        V_ = lambda Nd : eq.eq_pot_fermi(Eg/Vt,Vi/Vt,Nd,Nc,Nv,ni)
        V = np.vectorize(V_)(Nd)
    return V

def rhs_1d(V,Eg,Vi,Nc,Nv,Nd,stat):
    if(stat == 'boltzmann'):
        R,R_ = 2*np.sinh(V) - Nd, 2*np.cosh(V)
    else:
        V_ = V - Vi
        R,R_ = Nc*eq.F(V_) - Nv*eq.F(-V_-Eg) - Nd, Nc*eq.F_(V_) + Nv*eq.F_(-V_-Eg)
    return R-V*R_,R_

def update_V_adi(V,b,lap_x,lap_y,lap,Eg,Vi,Nc,Nv,Nd,ni,iterations,stat='boltzmann'):
    Nc,Nv,ND = Nc/ni,Nv/ni,Nd/ni
    V_i = np.copy(V)
    if(stat == 'boltzmann'):
        for k in range(iterations):
            for i in range(2):
                lx,ly,l,bc = lap_x[:,i::2,:],lap_y[:,i::2,:],lap[i::2,:],b[:,i::2,:]
                cur_V = V[i+1:-1:2,1:-1]
                R,R_ = 2*np.sinh(cur_V) - ND[i::2,:], 2*np.cosh(cur_V)
                r = bc[0]*(R - cur_V*R_ - bc[1]) + (1-bc[0])*cur_V - (ly[0]*V[i:-2:2,1:-1] + ly[1]*V[i+2::2,1:-1])
                V[i+1:-1:2,1:-1] = solve_tdma(lx[0].T,(l-bc[0]*R_).T,np.copy(lx[1].T),r.T).T
            
            for i in range(2):
                lx,ly,l,bc = lap_x[:,:,i::2],lap_y[:,:,i::2],lap[:,i::2],b[:,:,i::2]
                cur_V = V[1:-1,i+1:-1:2]
                R,R_ = 2*np.sinh(cur_V) - ND[:,i::2], 2*np.cosh(cur_V)
                r = bc[0]*(R - cur_V*R_ - bc[1]) + (1-bc[0])*cur_V - (lx[0]*V[1:-1,i:-2:2] + lx[1]*V[1:-1,i+2::2])
                V[1:-1,i+1:-1:2] = solve_tdma(ly[0],l-bc[0]*R_,np.copy(ly[1]),r) 
    else:
        V -= Vi
        for k in range(iterations):
            for i in range(2):
                lx,ly,l,bc = lap_x[:,i::2,:],lap_y[:,i::2,:],lap[i::2,:],b[:,i::2,:]
                cur_V = V[i+1:-1:2,1:-1]
                R,R_ = Nc*eq.F(cur_V) - Nv*eq.F(-cur_V-Eg) - ND[i::2,:], Nc*eq.F_(cur_V) + Nv*eq.F_(-cur_V-Eg)
                r = bc[0]*(R - cur_V*R_ - bc[1]) + (1-bc[0])*cur_V - (ly[0]*V[i:-2:2,1:-1] + ly[1]*V[i+2::2,1:-1]) 
                V[i+1:-1:2,1:-1] = solve_tdma(lx[0].T,(l-bc[0]*R_).T,np.copy(lx[1].T),r.T).T
            for i in range(2):
                lx,ly,l,bc = lap_x[:,:,i::2],lap_y[:,:,i::2],lap[:,i::2],b[:,:,i::2]
                cur_V = V[1:-1,i+1:-1:2]
                R,R_ = Nc*eq.F(cur_V) - Nv*eq.F(-cur_V-Eg) - ND[:,i::2], Nc*eq.F_(cur_V) + Nv*eq.F_(-cur_V-Eg)
                r = bc[0]*(R - cur_V*R_ - bc[1]) + (1-bc[0])*cur_V - (lx[0]*V[1:-1,i:-2:2] + lx[1]*V[1:-1,i+2::2])
                V[1:-1,i+1:-1:2] = solve_tdma(ly[0],l-bc[0]*R_,np.copy(ly[1]),r) 
        V += Vi
    dV = np.abs(V-V_i)
    err = dV/np.maximum(1,abs(V)) 
    return V,np.amax(err)

def update_V(V,b,lap_x,lap_y,lap,Eg,Vi,Nc,Nv,Nd,ni,w,iterations,stat='boltzmann'):
    Nc,Nv,ND = Nc/ni,Nv/ni,Nd/ni
    order = [[0,0],[1,1],[0,1],[1,0]]
    V_i = np.copy(V)
    if(stat == 'boltzmann'):
        for k in range(iterations):
            for it in range(4):
                i,j = order[it]
                lx,ly,l,bc = lap_x[:,i::2,j::2],lap_y[:,i::2,j::2],lap[i::2,j::2],b[:,i::2,j::2]
                Vy,Vx,cur_V = V[:,j+1:-1:2],V[i+1:-1:2,:],V[i+1:-1:2,j+1:-1:2]
                r = ly[0]*Vy[i:-2:2,:] + ly[1]*Vy[i+2::2,:] + lx[0]*Vx[:,j:-2:2] + lx[1]*Vx[:,j+2::2] + l*cur_V 
                R,R_ = 2*np.sinh(cur_V) - ND[i::2,j::2], 2*np.cosh(cur_V)
                cur_V += w*bc[0]*(R - bc[1] - r) / (l - R_)
    else:
        V -= Vi
        for k in range(iterations):
            for it in range(4):
                i,j = order[it]
                lx,ly,l,bc = lap_x[:,i::2,j::2],lap_y[:,i::2,j::2],lap[i::2,j::2],b[:,i::2,j::2]
                Vy,Vx,cur_V = V[:,j+1:-1:2],V[i+1:-1:2,:],V[i+1:-1:2,j+1:-1:2]
                r = ly[0]*Vy[i:-2:2,:] + ly[1]*Vy[i+2::2,:] + lx[0]*Vx[:,j:-2:2] + lx[1]*Vx[:,j+2::2] + l*cur_V 
                R,R_ = Nc*eq.F(cur_V) - Nv*eq.F(-cur_V-Eg) - ND[i::2,j::2], Nc*eq.F_(cur_V) + Nv*eq.F_(-cur_V-Eg)
                cur_V += w*bc[0]*(R - bc[1] - r) / (l - R_)
        V += Vi
    dV = np.abs(V-V_i)
    err = dV/np.maximum(1,abs(V)) 
    return V,np.amax(err)

def update_V_1d(V,lap,d,Eg,Vi,Nc,Nv,Nd,ni,stat='boltzmann',iterations=10):
    Nc,Nv,ND = Nc/ni,Nv/ni,Nd/ni
    Vc = np.copy(V)
    for i in range(iterations):
        R,R_ = rhs_1d(V,Eg,Vi,Nc,Nv,ND,stat)
        V = solve_tdma(np.copy(lap[0]),lap[1]-R_,np.copy(lap[2]),R-d)
    dV = abs(Vc - V)
    return V,max(dV/np.maximum(1,abs(V)))

def normalise_boundary_1d(bc,Vt,L):
    for i in range(2):
        bc[i][1] /= Vt
        if(bc[i][0]=='n'):
            bc[i][1] *= L
    return bc

def normalise_boundary_2d(bc_top,bc_bot,Vt,L):
    N = len(bc_top)
    for i in range(N):
        bc_top[i][1]/=Vt
        bc_bot[i][1]/=Vt
        if(bc_top[i][0]=='n'):
            bc_top[i][1]*=L
        if(bc_bot[i][0]=='n'):
            bc_bot[i][1]*=L
    return bc_top,bc_bot
