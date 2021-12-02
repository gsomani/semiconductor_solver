import numpy as np

def solve_tdma(a,b,c,d):
    N=len(d)
    for i in range(N):
        b[i] -= c[i-1]*a[i]
        c[i] /= b[i]
        d[i] = (d[i]-d[i-1]*a[i])/b[i]
    for i in range(N-2,-1,-1):
        d[i] -= c[i]*d[i+1]
    return d
