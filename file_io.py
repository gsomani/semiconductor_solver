import numpy as np

superscript_2 = '\u00B2'
superscript_3 = '\u00B3'
mu = '\u03BC'
unit = [mu+'m', 'Volts', '/cm'+ superscript_3, 'A/cm'+ superscript_2]
var = ['Nd(%s)'% unit[2],'V(%s)' % unit[1],'n(%s)'% unit[2],'p(%s)'% unit[2]]
J_var = ['Jn(%s)'% unit[3],'Jp(%s)' % unit[3]]

def load_solution_1d(filename,current=False):
    file = open(filename,'r')
    N = int(file.readline().split()[1])
    x = np.array(file.readline().split(),dtype=float)
    l = file.readline().split()
    c,l = int(l[0]),l[1:]
    if( len(x)!=N or c!= len(l)):
        print('File not valid')
        return
    var_list = l[1:]
    if(current):
        svar = var + J_var
    else:
        svar = var
    var_index = [var_list.index(v) for v in svar]
    data = np.empty([N,c-1])
    for i in range(N):
        l = file.readline().split()
        nx = [int(l[0])]
        data[nx] = np.array(l[1:])
    file.close()
    d = [data[:,i] for i in var_index]
    return x,d

def load_solution_2d(filename):
    file = open(filename,'r')
    geom = file.readline().split()[0]
    N = int(file.readline().split()[1])
    x = np.array(file.readline().split(),dtype=float)
    M = int(file.readline().split()[1])
    y = np.array(file.readline().split(),dtype=float)
    l = file.readline().split()
    c,m,l = int(l[0]),int(l[1]),l[2:]
    if( len(x)!=N or len(y)!=M or c!= len(l) or m != M*N):
        print('File not valid')
        return
    var_list = l[2:]
    var_index = [var_list.index(v) for v in var]
    data = np.empty([M,N,c-2])
    for i in range(M):
        for j in range(N):
            l = file.readline().split()
            nx,ny = [int(l[0]),int(l[1])]
            data[ny][nx] = np.array(l[2:])
    file.close()
    d = [data[:,:,i] for i in var_index] 
    return [geom,x,y],d

def write_jv_log(V,J_left,J_right,J,filename):
    file = open(filename,'w')
    print("\nWriting J-V to %s" % (filename))
    file.write('%i\n' % len(V))
    file.write('V(%s) Jn_left(%s) Jp_left(%s) Jn_right(%s) Jp_right(%s) J(%s)' % (unit[0],unit[3],unit[3],unit[3],unit[3],unit[3]))
    for i in range(len(V)):
        file.write('\n%f %E %E %E %E %E' %(V[i],J_left[i,0],J_left[i,1],J_right[i,0],J_right[i,1],J[i]))
    file.close()

def write_solution_1d(mesh,Nd,sol,filename,current=False):
    file = open(filename,'w')
    x = mesh[1]
    print("\nWriting mesh and solution to %s" % (filename))
    file.write(mesh[0])
    file.write('\nx(%s) %i\n' % (unit[0],len(x)))
    x.tofile(file," ", "%E")
    if(current):
        svar = var + J_var
    else:
        svar = var
    file.write('\n%i \nnx' % (1+len(svar)))
    for v in svar:
        file.write('\t %s' % v)
    for i in range(len(x)):
        file.write('\n%i %E ' %(i,Nd[i]))
        for s in sol:
            file.write('%E ' % s[i])
    file.close()

def write_solution_2d(mesh,Nd,sol,filename):
    file = open(filename,'w')
    m = ['x','y']
    print("\nWriting mesh and solution to %s" % (filename))
    file.write(mesh[0])
    for i in range(1,3):
        file.write('\n%s(%s) %i\n' % (m[i-1],unit[0],len(mesh[i])))
        mesh[i].tofile(file," ", "%E")
    M,N = sol[0].shape
    file.write('\n%i %i\nnx ny ' % (2+len(var),M*N))
    for v in var:
        file.write('\t %s' % v)
    for i in range(M):
        for j in range(N):
            file.write('\n%i %i %E ' %(j,i,Nd[i][j]))
            for s in sol:
                file.write('%E ' % s[i][j])
    file.close()
