import numpy as np
import bc,mesh,eq,init,deriv
from scipy.interpolate import RectBivariateSpline
from file_io import superscript_3
from eq import exact_F

F = np.vectorize(eq.exact_F)

def solver(constants,geom,x,y,V,Nd_n,bc_top,bc_bot,eps,iterations,damp=1.05,w=1.85,s='boltzmann',mg=True):
    ni,Nc,Nv = constants.ni,constants.Nc,constants.Nv
    Vt,Eg,Vi = constants.Vt,constants.Eg,constants.Vi
 
    Eg_n,Vi_n = Eg/Vt,Vi/Vt
    Nc_n,Nv_n = Nc/ni,Nv/ni

    V_ = np.pad(V,((1,1),(1,1)),mode='reflect')  # Pad mesh
    lap_x,lap_y,lap,b= deriv.laplacian_2d(y,x,bc_top,bc_bot,geom) # laplacian stencil   
    stat = ['boltzmann']    
    if(s=='fermi'):
        stat.append('fermi')

    for st in stat:
        dV=dV_cur=1
        if(mg):            
            while(dV>=eps):
                V_,dV = init.update_V(V_,b,lap_x,lap_y,lap,Eg_n,Vi_n,Nc_n,Nv_n,Nd_n,1,w,iterations,st)
                if(dV_cur<dV):
                    w /= damp
                dV_cur=dV
        else:        
            while(dV>=eps):
                V_,dV = init.update_V_adi(V_,b,lap_x,lap_y,lap,Eg_n,Vi_n,Nc_n,Nv_n,Nd_n,1,iterations,st)
    print("Solution converged for %i x %i grid" %(len(x),len(y)))

    return V_[1:-1,1:-1]

def multigrid(geom,x,y,V,Nd_n,bc_top,bc_bot,level,eps,damp,w,iterations):
    if(len(x)*len(y)<1000 or len(x)<20 or len(y)<20):
        return solver(geom,x,y,V,Nd_n,bc_top,bc_bot,eps,iterations,damp,w)
    V = multigrid(geom,x[::2],y[::2],V[::2,::2],Nd_n[::2,::2],bc_top[::2],bc_bot[::2],level+1,eps/2,damp,w,iterations)
    f = RectBivariateSpline(y[::2],x[::2],V,kx=1,ky=1)
    V = f(y,x)
    if(level==0):
        set_dirichlet(V,x,bc_top,bc_bot)
        return solver(geom,x,y,V,Nd_n,bc_top,bc_bot,eps,iterations,damp,w,s='fermi') 
    return solver(geom,x,y,V,Nd_n,bc_top,bc_bot,eps,iterations,damp,w)

def set_dirichlet(V,x,bc_top,bc_bot):
    for i in range(len(x)):
        if(bc_top[i][0]=='d'):
            V[0][i] = bc_top[i][1]
        if(bc_top[i][0]=='d'):
            V[-1][i] = bc_bot[i][1]

def solve_2d(constants,mesh,Nd,bc_top,bc_bot,eps=2**-21,w=1.85,damp=1.05,iterations=10,mg=False):
    ni,Nc,Nv = constants.ni,constants.Nc,constants.Nv
    Vt,Eg,Vi = constants.Vt,constants.Eg,constants.Vi
 
    Eg_n,Vi_n = Eg/Vt,Vi/Vt
    Nc_n,Nv_n = Nc/ni,Nv/ni
    L_um = 1e4*constants.Ld_i

    # Normalising all variables
    geom = mesh[0]
    x,y = mesh[1]/L_um,mesh[2]/L_um
 
    bc_top,bc_bot = init.normalise_boundary_2d(bc_top,bc_bot,Vt,L_um)
    Nd_n = Nd/ni

    # Initial guess
    V = init.init(Eg_n,Vi_n,1,Nd_n,Nc_n,Nv_n,1)

    # Set points at boundary with dirichlet boundary conditions
    set_dirichlet(V,x,bc_top,bc_bot)

    # Solution iterations
    if(mg):
        V = multigrid(geom,x,y,V,Nd_n,bc_top,bc_bot,0,eps,damp,w,iterations)
    else:
        V = solver(constants,geom,x,y,V,Nd_n,bc_top,bc_bot,eps,iterations,s='fermi',mg=False)
    print("Solution converged")
    
    n = Nc*F(V-Vi_n)
    p = Nv*F(-V+Vi_n-Eg_n)
    V *= Vt

    return [V,n,p]

def solve_1d(constants,mesh,Nd,bc,eps=2**-40,st='fermi',print_log=True):
    
    ni,Nc,Nv = constants.ni,constants.Nc,constants.Nv
    Vt,Eg,Vi = constants.Vt,constants.Eg,constants.Vi
 
    Eg_n,Vi_n = Eg/Vt,Vi/Vt
    Nc_n,Nv_n = Nc/ni,Nv/ni
    L_um = 1e4*constants.Ld_i

    geom = mesh[0]
    x = mesh[1]/L_um
    Nd_n = Nd/ni

    bc = init.normalise_boundary_1d(bc,Vt,L_um)
    V = init.init(Eg_n,Vi_n,1,Nd_n,Nc_n,Nv_n,1,stat='boltzmann')

    V = np.array(V,dtype=np.longdouble)

    # Set points at boundary with dirichlet boundary conditions  
    if(bc[0][0]=='d'):
        V[0] = bc[0][1]
    if(bc[1][0]=='d'):
        V[len(x)-1] = bc[1][1]
    
    lap,d,[start,stop] = deriv.laplacian_tridiagnol(x,bc,geom) # Set up tridiagnol matrix
    
    s = ['boltzmann']
    if(st=='fermi'):
        s = s + ['fermi']
    
    for stat in s:
        dV=1
        while(dV>=eps):
            V[start:stop],dV = init.update_V_1d(V[start:stop],lap,d,Eg_n,Vi_n,Nc_n,Nv_n,Nd_n[start:stop],1,stat,iterations=10) 
    
    if(print_log):    
        print("Solution converged")

    # Converting to phsyical units
    if(st=='fermi'):
        n = Nc*F(V-Vi_n)
        p = Nv*F(-V+Vi_n-Eg_n)
    else:
        n = ni*np.exp(V)
        p = ni*np.exp(-V)

    V *= Vt

    return [V,n,p]

def mesh_1d(constants,nodes,doping,contact,geom='rect',stat='fermi'):
    ni,Nc,Nv = constants.ni,constants.Nc,constants.Nv
    Vt,Eg,Vi = constants.Vt,constants.Eg,constants.Vi
    
    loc,spac = [n[0] for n in nodes],[n[1] for n in nodes]
    x = mesh.gen_mesh_1d(loc,spac)[0]
    N = len(x)
    if(geom=='cyl'):
        print('Cylindrical structure')    
        contact = [['n',0],contact[1]]
    elif(geom=='sph'):
        print('Spherical structure')
        contact = [['n',0],contact[1]]
    else:
        print('Rectangular structure')
    print("Total number of nodes = %i" % N)
    dope = np.vectorize(doping)
    Nd = dope(x)
    print("\nMaximum net doping = %E /cm%s\nMinimum net doping = %E /cm%s\nMinimum absolute doping = %E /cm%s\n" % (np.amax(Nd),superscript_3,np.amin(Nd),superscript_3,np.amin(abs(Nd)),superscript_3))
    boundary = bc.set_eq_bc_1d(contact,Eg,Vi,Vt,Nd,Nc,Nv,ni,stat)
    return [geom,x],Nd,boundary

def mesh_2d(constants,nodes_x,nodes_y,doping,contact_top,contact_bot,geom='rect'):
    ni,Nc,Nv = constants.ni,constants.Nc,constants.Nv
    Vt,Eg,Vi = constants.Vt,constants.Eg,constants.Vi

    lx,sx = [n[0] for n in nodes_x],[n[1] for n in nodes_x]
    ly,sy = [n[0] for n in nodes_y],[n[1] for n in nodes_y]
    (x,nx),(y,ny) = mesh.gen_mesh_2d(lx,sx,ly,sy)
    M,N = len(y),len(x)
    if(geom=='cyl'):
        print('Cylindrical structure')
    else:
        print('Rectangular structure') 
    print("Number of nodes along x-direction = %i\nNumber of nodes along y-direction = %i\nTotal number of nodes = %i" % (N,M,N*M))
    top = []
    bot = []
    sections = len(nx)-1
    for i in range(sections):
        section_length = nx[i+1]-nx[i]
        top,bot = top + [contact_top[i]]*section_length, bot + [contact_bot[i]]*section_length
    top.append(contact_top[sections-1])
    bot.append(contact_bot[sections-1])
    dope = np.vectorize(doping)
    xx,yy = np.meshgrid(x,y)
    Nd = dope(xx,yy)
    print("\nMaximum net doping = %E /cm%s\nMinimum net doping = %E /cm%s\nMinimum absolute doping = %E /cm%s\n" % (np.amax(Nd),superscript_3,np.amin(Nd),superscript_3,np.amin(abs(Nd)),superscript_3))
    bc_top,bc_bot = bc.set_bc_2d(top,bot,Eg,Vi,Vt,Nd,Nc,Nv,ni)
    return [geom,x,y],Nd,bc_top,bc_bot
