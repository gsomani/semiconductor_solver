import numpy as np

def double_derivative_stencil(x):
    diff = [x[1]-x[0],x[2]-x[1],x[2]-x[0]]
    a = 2/(diff[0]*diff[2])
    c = 2/(diff[1]*diff[2])
    return np.array([a,c])

def double_derivative_stencil_cyl(x):
    diff = [x[1]-x[0],x[2]-x[1],x[2]-x[0]]
    d = 1/(x[1]*diff[2])    
    a = 2/(diff[0]*diff[2]) - d
    c = 2/(diff[1]*diff[2]) + d
    return np.array([a,c])

def double_derivative_stencil_sph(x):
    diff = [x[1]-x[0],x[2]-x[1],x[2]-x[0]]
    d = 1/(x[1]*diff[2])    
    a = 2/(diff[0]*diff[2]) - 2*d
    c = 2/(diff[1]*diff[2]) + 2*d
    return np.array([a,c])

def laplacian_tridiagnol(x,bc,geom):
    N = len(x)
    d = np.zeros(N,dtype=np.longdouble)
    start,stop = 0,0
    x_ = np.empty(N+2,dtype=np.longdouble)

    x_[0],x_[1:-1],x_[-1] = -x[1],x,2*x[-1]-x[-2]

    mat = np.empty([3,N])
    if(geom=='cyl'):
        mat[::2,0] = 2*double_derivative_stencil([x_[0],x_[1],x_[2]])
        mat[::2,1:] = double_derivative_stencil_cyl([x_[1:-2],x_[2:-1],x_[3:]])
    elif(geom=='sph'):
        mat[::2,0] = 3*double_derivative_stencil([x_[0],x_[1],x_[2]])
        mat[::2,1:] = double_derivative_stencil_sph([x_[1:-2],x_[2:-1],x_[3:]])
    else:
        mat[::2] = double_derivative_stencil([x_[:-2],x_[1:-1],x_[2:]])    

    mat[2,0] += mat[0,0]
    mat[0,-1] += mat[2,-1]
    mat[0,0] = mat[2,-1] = 0
    
    mat[1] = -(mat[0]+mat[2])

    if(bc[0][0]=='n'):
        d[0] = 2*bc[0,1]/(x[1]-x[0])        
    else:
        start = 1
        d[1] += mat[0,1]*bc[0][1]
        mat[0,1] = 0
    if(bc[1][0]=='n'):
        d[-1] = 2*bc[1,1]/(x[-2]-x[-1])
    else:
        stop -= 1
        d[-2] += mat[2,-2]*bc[1][1]
        mat[2,-2] = 0
    return mat[:,start:stop],d[start:stop],[start,stop]

def laplacian_2d(y,x,bc_top,bc_bot,geom):
    M,N = len(y),len(x)
    x_ = np.empty(N+2)        
    y_ = np.empty(M+2)
    x_[0],x_[1:-1],x_[-1] = -x[1],x,2*x[-1]-x[-2]
    y_[0],y_[1:-1],y_[-1] = -y[1],y,2*y[-1]-y[-2]

    lx,ly,l = np.zeros([2,M,N]),np.zeros([2,M,N]),np.zeros([M,N])
    lap_x = np.empty([2,N])
    b = np.zeros([2,M,N])
    b[0] = 1
    lap_y =  double_derivative_stencil([y_[:-2],y_[1:-1],y_[2:]])
    if(geom=='cyl'):
        lap_x[:,0] = 2*double_derivative_stencil([x_[0],x_[1],x_[2]])
        lap_x[:,1:] = double_derivative_stencil_cyl([x_[1:-2],x_[2:-1],x_[3:]])
    else:
        lap_x = double_derivative_stencil([x_[:-2],x_[1:-1],x_[2:]])
    
    lap_y[1,0] += lap_y[0,0]
    lap_y[0,-1] += lap_y[1,-1]
    
    lap_x[1,0] += lap_x[0,0]
    lap_x[0,-1] += lap_x[1,-1]

    lap_y[1,-1] = lap_y[0,0] = lap_x[1,-1] = lap_x[0,0] = 0

    lx[0],ly[0] = np.meshgrid(lap_x[0],lap_y[0])
    lx[1],ly[1] = np.meshgrid(lap_x[1],lap_y[1])
        
    for j in range(N):
        if(bc_top[j][0]=='n'):
            b[1,0,j]= 2*bc_top[j][1]/(y[1]-y[0])
        else:
            b[0,0,j] = 0
            ly[:,0,j] = lx[:,0,j] = 0
        if(bc_bot[j][0]=='n'):	
            b[1,-1,j] = 2*bc_bot[j][1]/(y[-2]-y[-1])
        else:
            b[0,-1,j] = 0
            ly[:,-1,j] = lx[:,-1,j] = 0

    l = -(lx[0]+lx[1]+ly[0]+ly[1]) + (1 - b[0])

    return lx,ly,l,b
