import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np

unit = ['$\mu m$', 'Volts', '$/cm^3$','$A/cm^2$']

def draw_mesh(mesh,show=True,style='solid',color='black',width=0.1,title=True):
    x,y = mesh[1:]
    plt.xlabel('x(%s)' % unit[0])
    plt.ylabel('y(%s)' % unit[0])
    plt.ylim([max(y),min(y)])
    plt.xlim([min(x),max(x)])
    plt.hlines(y, x[0], x[-1],linestyles=style,linewidth = width, color = color)
    plt.vlines(x, y[0], y[-1],linestyles=style,linewidth = width, color = color)
    plt.gca().set_aspect('equal')
    if(title):
        plt.title('Mesh')
    if(show):
        plt.show()

def contour_plot(mesh,data,name,dim,ni=1,scale='linear',mesh_draw=False,grid=True,show=True,filename=None,levels=256,cmap=None):
    x,y = mesh[1:]
    plt.xlabel('x(%s)' % unit[0])
    plt.ylabel('y(%s)' % unit[0])
    norm=None
    ticks=None
    extend='neither'
    ni_level = int(np.floor(np.log10(ni)))
    max_level = int(2*np.ceil(np.log10(ni)))

    if(mesh_draw):
        draw_mesh(mesh,show=False,title=False)
    
    def extend_check(data,levels):
        c = [np.amin(data)-levels[0],np.amax(data)-levels[-1]]
        if(c[0]<0 and c[1]>0):
            extend='both'
        elif(c[1]>0):
            extend='max'
        elif(c[0]<0):
            extend='min'
        else:
            extend='neither'
        return extend
    
    if(scale=='log'):
        norm=colors.LogNorm()
        levels = np.logspace(0,max_level,levels)
        ticks = np.logspace(0,max_level,max_level+1)
        extend=extend_check(data,levels)
    elif(scale=='sym_log'):
        lp = np.logspace(ni_level-1,max_level,levels+1)
        levels = np.concatenate([-lp[::-1],lp])
        tr = np.logspace(ni_level,max_level,max_level-ni_level+1)
        ticks = np.concatenate([-tr[::-1],[0],tr])
        norm=colors.SymLogNorm(linthresh=ni/10, linscale=1,base=10)
        extend=extend_check(data,levels)

    if(dim=='volt'):
        u = unit[1]
    else:
        u = unit[2]

    plt.xlim([min(x),max(x)])
    plt.ylim([max(y),min(y)])
    plt.contourf(x,y,data, norm=norm, levels=levels,cmap=cmap,extend=extend)
    plt.colorbar(ticks=ticks).set_label("%s (%s)" %(name,u))
    plt.grid(grid, which="both")
    plt.gca().set_aspect('equal')
    plt.title(name)
    if(filename != None):
        plt.savefig(filename)
    if(show):
        plt.show()
    plt.close()

def plot_1d(mesh,data,name,dim,ni=1,scale='linear',grid=True,show=True,filename=None):
    x = mesh[1]
    if(scale=='logy'):
        plt.yscale('log')
    elif(scale=='logx'):
        plt.xscale('log')
    elif(scale=='loglog'):
        plt.xscale('log')
        plt.yscale('log')
    elif(scale=='symlog'):
        plt.yscale('symlog',linthresh=ni)
    plt.plot(x,data)
    plt.grid(grid, which="both")
    plt.title(name)
    if(dim=='volt'):
        u = unit[1]
    elif(dim=='current density'):
        u = unit[3]
    else:
        u = unit[2]
    plt.ylabel('%s (%s)' %(name,u))
    plt.xlabel('x($\mu m$)')
    if(filename != None):
        plt.savefig(filename)
    if(show):
        plt.show()
    plt.close()

def plot_jv_log(V,J,name,J0,scale='linear',show=True,filename=None):
    if(scale=='log'):
        plt.yscale('log')
    elif(scale=='symlog'):
        plt.yscale('symlog',linthresh=J0/10)
    plt.plot(V,J)
    plt.grid(True, which="both")
    plt.title(name)
    plt.ylabel('J (%s)' % unit[3])
    plt.xlabel('V (%s)' % unit[1])
    if(filename != None):
        plt.savefig(filename)
    if(show):
        plt.show()
    plt.close()
