import numpy as np
from constants import discrete_K
from tqdm import tqdm

# p_l and p_r -- pressure on borders, functions of z and t
# p_l and p_r -- pressure everywhere when t=0, a functions of x and z
class NonStationaryGrid():
    def __init__(self, p_l, p_r, p_0, n=100, m=50, t=1000):
        
        self.n = n
        self.m = m
        self.t = t
        self.m_ = 2e2
        self.beta = 1e2
        
        self.hx = 1/n
        self.hz = 1/m
        self.ht = 1/t

        self.p = np.zeros((t, n, m))

        t = np.linspace(0, 1, t)
        x = np.linspace(0, 1, n)
        z = np.linspace(0, 1, m)
        tz, zt = np.meshgrid(t, z)

        # p[t, x, z]
        self.p[:,0,:] = np.vectorize(p_l)(tz, zt).T
        self.p[:,n-1,:] = np.vectorize(p_r)(tz, zt).T
        self.p[0, :, :] = p_0
        
        self.K = self.K = discrete_K(n=n, m=m)

        

    def iterate(self, K=None):
      if K is None:
        K=self.t
      for _ in tqdm(range(K-1)):
        self.step(_)
    
    def step(self, t):
        for i in range(1, self.n-1):
            for j in range(1, self.m - 1):
              vxx = (self.K[i+1,j]*(self.p[t, i+1,j]-self.p[t,i,j]) - self.K[i,j]*(self.p[t, i,j]-self.p[t, i-1,j]) )/(self.hx * self.hx)
              vzz = (self.K[i,j+1]*(self.p[t, i,j+1]-self.p[t,i,j]) - self.K[i,j]*(self.p[t, i,j]-self.p[t, i,j-1]) )/(self.hz * self.hz)
              self.p[t+1, i, j] = self.p[t, i, j] + self.ht * (vzz + vxx) / (self.m_ * self.beta)
              # print(self.p[t, i, j], self.ht * (vzz + vxx), self.p[t+1, i, j])
        self.p[t+1, :, 0] = self.p[t+1, :, 1]
        self.p[t+1, :, -1] = self.p[t+1, :, -2]
