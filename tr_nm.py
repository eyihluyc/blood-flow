import numpy as np
from constants import discrete_K
from tqdm import tqdm


class TransientGrid():
    def __init__(self, p_l, p_r, p_0, n, m, t):
        
        self.n = n
        self.m = m
        self.t = t

        self.m_ = 10
        self.beta = 1
        self.mu = 3
        
        self.hx = 1/n
        self.hz = 1/m
        self.ht = 1/t

        self.p = np.zeros((t, n, m))

        self.p[:,0,:] = p_l
        self.p[:,n-1,:] = p_r
        self.p[0, :, :] = p_0
        
        self.K = self.K = discrete_K(n=n, m=m)
        self.alpha = self.ht/(self.mu * self.m_ * self.beta)
        

    def iterate(self, eps, K=None):
      if K is None:
        K=self.t
      for _ in tqdm(range(K-1)):
        self.step(_, eps=eps)
    
    def step(self, t, eps):

        while True:
            prev = self.p.copy()
            
            for i in range(1, self.n-1):
                for j in range(1, self.m - 1):
                    num = self.p[t, i, j] + self.alpha * ((self.K[i+1,j] * self.p[t+1, i+1, j] + self.K[i,j] * self.p[t+1, i-1, j])/(self.hx * self.hx) + \
                                                          (self.K[i,j+1] * self.p[t+1, i, j+1] + self.K[i,j] * self.p[t+1, i, j-1])/(self.hz * self.hz))

                    denom = 1 + self.alpha * ((self.K[i+1, j] + self.K[i, j])/(self.hx * self.hx) + \
                                              (self.K[i, j+1] + self.K[i, j])/(self.hz * self.hz))

                    self.p[t+1, i, j] = num / denom
                    
            self.p[t+1, :, 0] = self.p[t+1, :, 1]
            self.p[t+1, :, -1] = self.p[t+1, :, -2]
            if (np.abs((self.p - prev)).max() < eps):
                break