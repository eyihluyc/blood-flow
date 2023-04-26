import numpy as np

# p_l and p_r -- left & right boundaries, s.t. their derivative at 0 and 1 is 0.
p_l = lambda z: 1 - np.cos(z * 2*np.pi)
p_r = lambda z: 1 - np.cos(z * 2*np.pi)


# an equation that satisfies boundary conditions
def BC(x, z):
  return (1 - x) * p_l(z) + x * p_r(z)
