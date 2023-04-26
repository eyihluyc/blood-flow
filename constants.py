import numpy as np

mu = 4 # cP; blood viscocity ranges from 3.5 to 5.5 cp

L = 0.01 # m; length of blood tube we consider
D = 2.7e-3 # m; diameter of an artery; ranges from 2.54 to 2.85 mm

def K(x, z): # permeability drawn according to the scheme from notes
    
    clot_radius = 0.8 * (D/2)
    clot_center_x, clot_center_z = L/2, 0
    
    # due to symmetry:
    z = min(z, D-z)
    
    # distance from point (x, z) to the clot center
    dist = np.linalg.norm((x - clot_center_x, z - clot_center_z))
    
    if dist < clot_radius:
        ans = 1e-15
    else:
        ans = 1

    return ans

def K_scaled(x, z):
  return K(x/L, z/D)

# scaled
def dKdx(x, z, eps=L/100):
  return (K_scaled(x+eps, z) - K_scaled(x-eps, z)) / (2*eps)

def dKdz(x, z, eps=D/100):
  return (K_scaled(x, z+eps) - K_scaled(x, z-eps)) / (2*eps)