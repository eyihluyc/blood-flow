import numpy as np
import pandas as pd

cardiac_cycle = pd.read_csv("pressure.csv").to_numpy()
# N = 30
# M = 30
# T = 10000

# cardiac_cycle_interp = np.interp(np.linspace(0, 1, T), cardiac_cycle[:, 0], cardiac_cycle[:, 1], period=1)
# p_l = np.broadcast_to(cardiac_cycle_interp, (M, T)).T
# p_r = np.broadcast_to(cardiac_cycle_interp * 0.99, (M, T)).T

# p_l and p_r -- left & right boundaries, s.t. their derivative at 0 and 1 is 0.
p_l = lambda z: 120
p_r = lambda z: 115


# an equation that satisfies boundary conditions
def BC(x, z):
  return (1 - x) * p_l(z) + x * p_r(z)
