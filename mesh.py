import numpy as np

def gp(s,l):
    r = (l-s[0])/(l-s[1])
    R = s[1]/s[0]
    if(R==1):
        n = np.round(l/s[0])
        a = l/n
    else:    
        n = np.round(np.log(R)/np.log(r) + 1)
        a = l*(r-1)/(r**n-1)
    return a,r,int(n)

def interpolate_mesh(a,r,n):
    mesh = np.linspace(a,n*a,n)
    if(r==1):
        return mesh
    for i in range(2,n+1):
        mesh[i-1]=a*(r**i-1)/(r-1)
    return mesh

def gen_mesh_parameters(loc,spac):
    sections = len(loc) - 1
    n = np.empty(sections,dtype=int)
    a = np.empty(sections)
    r = np.empty(sections)
    for i in range(sections):   
        a[i],r[i],n[i] = gp([spac[i],spac[i+1]],loc[i+1]-loc[i]) 
    return a,r,n

def gen_mesh_1d(loc,spac):
    a,r,n = gen_mesh_parameters(loc,spac)
    N = np.sum(np.array(n))
    sections = len(n)
    mesh = np.zeros(N+1)
    cur=1
    mesh[0]=loc[0]
    node_num = [0]
    for i in range(sections):
        mesh[cur:cur+n[i]] = loc[i] + interpolate_mesh(a[i],r[i],n[i])
        cur += n[i]
        node_num.append(cur-1)
        mesh[cur-1]=loc[i+1]
    mesh = np.array(mesh,dtype=np.longdouble)
    return mesh,node_num

def gen_mesh_2d(lx,sx,ly,sy):
    return gen_mesh_1d(lx,sx),gen_mesh_1d(ly,sy)
