import numpy as np
import non_eq,deriv
import init
from eq_solver import solve_1d,mesh_1d
from file_io import superscript_2
from scipy.interpolate import interp1d

def mesh(constants,nodes,doping,boundary):
    mesh,Nd,bc = mesh_1d(constants,nodes,doping,boundary,stat='boltzmann')
    return mesh,Nd,bc

def zero(x):
    return 0

def solve(x,Nd,tp,tn,mu,w,guess,g,glimit=0,eps=2**-36):
    V,fn,fp = guess    
    n = np.exp(V-fn)
    p = np.exp(-V+fp)
    dx = x[1:]-x[:-1]
    dx_inter = (x[2:]-x[:-2])/2

    lap,d,[start,stop] = deriv.laplacian_tridiagnol(x,[['d',V[0]],['d',V[-1]]],geom='rect') # Set up tridiagnol matrix

    V,n,p,fn,fp,err_V,err_n,err_p = non_eq.update_Vnp_gummel(dx_inter,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu,g,iterations=5)                                 
    err = max(err_V,err_n,err_p)

    cur = -1
    while(err>=eps):
        last_err = err
        if(cur==-1):
            V,n,p,fn,fp,err_V,err_n,err_p = non_eq.update_Vnp_newton(dx_inter,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu,g)
        else:
            V,n,p,fn,fp,err_V,err_n,err_p = non_eq.update_Vnp_gummel(dx_inter,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu,g,iterations=5)                                 
        err = max(err_V,err_n,err_p)
        if(last_err<err):
            cur *= -1
  
    V_ = (V[1:]+V[:-1])/2
    x_inter = (x[1:]+x[:-1])/2
    
    jn = (np.exp(V_-fn[1:]) - np.exp(V_-fn[:-1]))/dx
    jp = -mu*(np.exp(-V_+fp[1:]) - np.exp(-V_+fp[:-1]))/dx
    
    jn = interp1d(x_inter,jn,fill_value='extrapolate')
    jp = interp1d(x_inter,jp,fill_value='extrapolate')
    
    Jn = jn(x)
    Jp = jp(x)    
    
    J = (Jn[0]+Jp[0]+Jn[-1]+Jp[-1])/2

    dV,dfn,dfp = non_eq.update_Vnp_ac(dx_inter,dx,V,n,p,fn,fp,tp,tn,lap,d,Nd,mu,w,g)
    
    return np.array([V,fn,fp]),Jn,Jp,J

def generate_JV(constants,mesh,Nd,bc,w_MHz,bias,step=1,g=zero,eps=2**-30):
    t0_us, J0, R0 = constants.t0_us, constants.J0, constants.R0
    Vt, L_um, ni = constants.Vt, 1e4*constants.Ld_i, constants.ni
    mu_p, mu_n  = constants.mu_p, constants.mu_n
    tau = [constants.tp, constants.tn]

    generation = np.vectorize(g)
    gen = generation(mesh[1])/constants.R0
    tp = tau[0]/t0_us
    tn = tau[1]/t0_us
    w = w_MHz*t0_us
    x = mesh[1]/L_um
    Nd_n = Nd/ni
    mu = mu_p/mu_n

    source_current = np.dot(gen[1:-1],(x[2:]-x[:-2])/2)

    V_eq,n,p = solve_1d(constants,mesh,Nd,bc,eps=2**-40,st='boltzmann',print_log=False)
    solution_eq = np.array([V_eq/Vt,np.zeros(len(x)),np.zeros(len(x))])

    V = [0]
    r = 1 - x/x[-1]
    bias = np.array(bias)/Vt

    if(source_current==0):
        solution_i = solution_eq
        j_i=0
        jn_i = jp_i = np.zeros(len(x))
    else:    
        solution_i,jn_i,jp_i,j_i = solve(x,Nd_n,tp,tn,mu,solution_eq,gen,eps=2**-40)    

    J = [j_i*J0]
    J_left = [np.array([jn_i[0],jp_i[0]])*J0]
    J_right = [np.array([jn_i[-1],jp_i[-1]])*J0]

    print("Generation source current = %.3E A/cm%s" %(source_current*J0,superscript_2))
    print("\nV(Volts)\tJ_left(A/cm%s)\t\tJ_right(A/cm%s)\t\tJ(A/cm%s)"% (superscript_2,superscript_2,superscript_2))
    print("\t\tJn\tJp\t\tJn\tJp")
    print('---------------------------------------------------------------------------')
    print("%.3f\t\t%.3E %.3E\t%.3E %.3E\t%.3E" %(0,J_left[0][0],J_left[0][1],J_right[0][0],J_right[0][1],J[0]))
    
    def bias_sweep(solution_i,w,bias):
        n = int(np.ceil(abs(bias)))
        if(n>0):
            v = bias/n
            guess = solution_i + v*r
        last_solution = np.copy(solution_i)
        jn,jp,j = jn_i,jp_i,j_i
        for i in range(n):
            solution,jn,jp,j = solve(x,Nd_n,tp,tn,mu,w,guess,gen,eps=eps)
            J.append(j*J0)
            J_left.append(np.array([jn[0],jp[0]])*J0)
            J_right.append(np.array([jn[-1],jp[-1]])*J0)
            V.append(v*Vt)
            guess = 2*solution - last_solution
            last_solution = np.copy(solution)
            print("%.3f\t\t%.3E %.3E\t%.3E %.3E\t%.3E" %(V[-1],J_left[-1][0],J_left[-1][1],J_right[-1][0],J_right[-1][1],J[-1]))
            v = guess[1][0]
        return last_solution,jn,jp

    last_solution,jn,jp = bias_sweep(solution_i,w,bias[0])

    solution_start = [last_solution[0]*Vt,ni*np.exp(last_solution[0]-last_solution[1]),ni*np.exp(-last_solution[0]+last_solution[2]),jn*J0,jp*J0]    

    V = V[::-1]
    J = J[::-1]
    J_left = J_left[::-1]
    J_right = J_right[::-1]
    v = 0

    last_solution,jn,jp = bias_sweep(solution_i,w,bias[1])

    solution_end = [last_solution[0]*Vt,ni*np.exp(last_solution[0]-last_solution[1]),ni*np.exp(-last_solution[0]+last_solution[2]),jn*J0,jp*J0]
    
    return solution_start,solution_end,np.array(V),np.array(J_left),np.array(J_right),np.array(J)
