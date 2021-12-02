'''
gen_constants - generates constants
eq_solver - defines mesh and solves poisson eqaution
file_io - writes solution to file
plotter - plots solution
'''
from gen_constants import *
import file_io,plotter,eq_solver,current_solver
import matplotlib.pyplot as plt
from matplotlib import cm

def step(x,a):
    if(x>=a):
        return 1
    return 0

def boxcar(x,I):
    return step(x,I[0])-step(x,I[1])

def solution_eq_1d(params,nodes,doping,boundary,geom,solution_file="solution.out"):
    constants = material(params)
     
    # generates mesh and calculates boundary potential and fiels as required
    mesh,Nd,bc = eq_solver.mesh_1d(constants,nodes,doping,boundary)
    
    '''
    solve poisson eqaution
    Output units:
    Potential - volt
    electron and hole density - per cubic centimeter 
    '''
    solution = eq_solver.solve_1d(constants,mesh,Nd,bc)
    V,n,p = solution

    # writes solution to file
    file_io.write_solution_1d(mesh,Nd,solution,solution_file)

    return [mesh,Nd,V,n,p]

def plot_doping_1d(mesh,Nd,ni,show=True):
    plotter.plot_1d(mesh,Nd,'Net doping','density',ni=ni,scale='symlog',show=show)

def plot_solution_1d(mesh,Nd,V,n,p,params,show=True):
    constants = material(params)
    ni = constants.ni

    plot_doping_1d(mesh,Nd,ni=ni,show=show)
    plotter.plot_1d(mesh,V,'Potential','volt',filename='equilibrium_Potential_1d.png',show=show)
    plotter.plot_1d(mesh,n,'Electron density','density',scale='logy',filename='equilibrium_electrons_1d.png',show=show)
    plotter.plot_1d(mesh,p,'Hole density','density',scale='logy',filename='equilibrium_holes_1d.png',show=show)
    plotter.plot_1d(mesh,n+p,'Total carrier density','density',scale='logy',filename='total_carriers_equilibrium.png',show=show)
    plotter.plot_1d(mesh,p+Nd-n,'Net charge carrier density','density',ni=ni,scale='symlog',filename='net_charge_equilibrium.png',show=show)

def solution_eq_2d(params,nodes,doping,contact,geom,solution_file="solution.out"):
    constants = material(params)
    # generates mesh and calculates boundary potential and fiels as required
    mesh,Nd,bc_top,bc_bot = eq_solver.mesh_2d(constants,nodes[0],nodes[1],doping,contact[0],contact[1],geom)
    ''' 
    solve poisson eqaution
    Output units:
    Potential - volt
    electron and hole density - per cubic centimeter 
    '''
    solution = eq_solver.solve_2d(constants,mesh,Nd,bc_top,bc_bot)

    # writes solution to file
    file_io.write_solution_2d(mesh,Nd,solution,solution_file)

    V,n,p = solution

    return [mesh,Nd,V,n,p]

def plot_solution_2d(mesh,Nd,V,n,p,params,show=True):
    constants = material(params)
    ni = constants.ni
    # plots mesh
    plotter.draw_mesh(mesh,show=show)

    # plots doping profile
    plotter.contour_plot(mesh,Nd, 'Net doping','density',ni=ni, scale='sym_log',levels=64,cmap=cm.gist_rainbow,show=show)

    plotter.contour_plot(mesh,V,'Potential','volt',filename='equilibrium_potential_2d.png',cmap=cm.gist_rainbow,show=show)
    plotter.contour_plot(mesh,n,'Electron density','density',ni=ni,scale='log',filename='equilibrium_electrons_2d.png',levels=64,cmap=cm.gist_rainbow,show=show)
    plotter.contour_plot(mesh,p,'Hole density','density',ni=ni,scale='log',filename='equilibrium_holes_2d.png',cmap=cm.gist_rainbow,show=show)
    plotter.contour_plot(mesh,p+Nd-n, 'Net charge carrier density','density',ni=ni, scale='sym_log',filename='net_charge_equilibrium_2d.png',cmap=cm.gist_rainbow,show=show)
    plotter.contour_plot(mesh,p+n,'Total carrier density','density',ni=ni,scale='log',filename='carriers_equilibrium_2d.png',cmap=cm.gist_rainbow,show=show)

def gen(x):
    return 0

def solve_current(constants,nodes,doping,contact,bias,gen=gen): 
    params = material(constants)

    x,Nd,bc = current_solver.mesh(params,nodes,doping,contact) # define mesh and boundary conditions

    solution_start,solution_end,V,J_left,J_right,J = current_solver.generate_JV(params,x,Nd,bc,bias,gen) # calculate current

    file_io.write_jv_log(V,J_left,J_right,J,'pn_junction_jv.log') # write J-V data to dfile
   
    # write solutions at two bias points to files
    file_io.write_solution_1d(x,Nd,solution_start,"solution_reverse_bias.out",current=True)
    file_io.write_solution_1d(x,Nd,solution_end,"solution_forward_bias.out",current=True)

    return [x,Nd,solution_start,solution_end,V,J,J_left,J_right]

def plot_JV(V,J,params,show=True):
    constants = material(params)
    J0 = constants.J0
    plotter.plot_jv_log(V,J,'J-V (pn-juction)',J0,scale='linear',filename='jv_diode_linear.png',show=show) # plot J-V (linear)
    plotter.plot_jv_log(V,J,'J-V (pn-juction)',J0,scale='symlog',filename='jv_diode_log.png',show=show) # plot J-V

def plot_solution_bias(x,Nd,ni,solution,bias_type='',show=True):
    
    # plot solution
    V,n,p,Jn,Jp = solution

    plotter.plot_1d(x,V,'Potential','volt',filename='potential_'+ bias_type+'.png',show=show)
    plotter.plot_1d(x,n,'Electron density','density',scale='logy',filename='electron_'+ bias_type+'.png',show=show)
    plotter.plot_1d(x,p,'Hole density','density',scale='logy',filename='hole_'+ bias_type+'.png',show=show)
    plotter.plot_1d(x,n+p,'Total carrier density','density',scale='logy',show=show)
    plotter.plot_1d(x,p+Nd-n,'Net charge carrier density','density',ni=ni,scale='symlog',show=show)
    plotter.plot_1d(x,Jn,'Electron current density','current density',show=show)
    plotter.plot_1d(x,Jp,'Hole current density','current density',show=show)

def plot_solution_both_ends(x,Nd,params,solution_start,solution_end,show=True):
    
    constants = material(params)
    ni = constants.ni

    plot_doping_1d(x,Nd,ni,show=show)
    plot_solution_bias(x,Nd,ni,solution_start,bias_type='rb',show=show)
    plot_solution_bias(x,Nd,ni,solution_end,bias_type='fb',show=show)
