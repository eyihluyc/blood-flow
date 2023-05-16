import numpy as np

mu = 4 # cP; blood viscocity ranges from 3.5 to 5.5 cp

L = 0.01 # m; length of blood tube we consider
D = 2.7e-3 # m; diameter of an artery; ranges from 2.54 to 2.85 mm

# unit_scaled = True if input x, z are scaled from 0 to 1
def K(x, z, unit_scaled=True): # permeability drawn according to the scheme from notes
    
    if unit_scaled:
       x, z = x * L, z * D
    
    # due to symmetry:
    z = min(z, D-z)

    clot_center_x, clot_center_z = L/2, -D/15
    
    # distance from point (x, z) to the clot center
    dist = np.linalg.norm((x - clot_center_x, z - clot_center_z))
    
    return (1/(1+np.exp(-( 2000*dist ))))


def discrete_K(n=100, m=50):
  hx = 1/n
  hz = 1/m
  _K = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
        _K[i,j] = K(i*hx, j*hz, unit_scaled=True)
  return _K


# def dKdx(x, z, eps=L/100):
#   return (K(x+eps, z) - K(x-eps, z)) / (2*eps)

# def dKdz(x, z, eps=D/100):
#   return (K(x, z+eps) - K(x, z-eps)) / (2*eps)