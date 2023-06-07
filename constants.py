import numpy as np

# blood rheology constants

# mu = 5 # cP; blood viscocity ranges from 3.5 to 5.5 cp
mu = 5e-6 # 5e-3 Pa s = 5e-6 kg/mm/s
max_permeability = 0.283 # mm^2

# blood vessel geometry
L = 10 # mm; length of the artery we consider
D = 2.5 # mm; diameter of the artery

def get_permeability(center, radius, range=(1e-7, 1e-7)): 

    def permeability(x, z):
       # due to symmetry:
      z = min(z, D-z)

      center_x, center_z = center
      min_K, max_K = range
      
      # distance from point (x, z) to the clot center
      dist = np.linalg.norm(((x - center_x), z - center_z))
      
      return max_permeability if radius < dist else dist/radius * (max_K - min_K) + min_K

    return permeability


K1 = get_permeability((L/2, -1), 1.3) #  vessel with the resolving clot
K2 = get_permeability((L/2, 0.25), 0.4) #  vessel with clot at high risk of breakage
K3 = get_permeability((L/2, 0), 0.6, (1e-7, 0.1)) #  vessel with clot at high risk of breakage

def discretize_K(K, n, m):
  hx = L/n
  hz = D/m
  _K = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
        _K[i,j] = K(i*hx, j*hz)
  return _K
