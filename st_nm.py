from constants import discrete_K
import numpy as np

# finite difference grid:

# p_l and p_r -- np arrays of shape (m,)
class StationaryGrid():
    def __init__(self, p_l, p_r, n, m):
        self.n = n
        self.m = m
        
        self.hx = 1/n
        self.hz = 1/m

        self.p = np.zeros((n, m))
        self.p[0] = p_l
        self.p[-1] = p_r
        
        self.K = discrete_K(n=n, m=m)

    # initial values -- linear interpolation b/w left and right boundaries
        for j in range(0, m):
            self.p[:, j] = np.interp(np.arange(0, n), [0, n-1], [self.p[0, j], self.p[n-1, j]])
        
        self.iterations = 0
        self.eps = float('inf')
        self.max_div = float('inf')
        
    # apply gauss-seidel method with even/odd order until distance with L\inf is less than eps
    # or diverngence is bounded by max_div
    # or certain number of iterations is achieved
    def iterate(self, criterion, steps=100, eps=1e-5, max_div = 0.001):
        
        if (criterion == "iter"):
            for _ in range(steps):
                self.make_step()
        elif (criterion == "err"):
            while (self.eps > eps):
                self.make_step()
        elif (criterion == "div"):
            while (self.max_div > max_div):
                print(f"step {self.iterations}, max divergence is {self.max_div}")
                for _ in range(50):
                    self.make_step()
                self.calc_div()

                
    def make_step(self):

        prev = self.p.copy()
        
        self.iterations += 1
        self.half_step(0)
        self.half_step(1)
        
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]

        self.eps = np.max(np.abs(prev - self.p))
    
    def half_step(self, oddity):

        for i in range(1, self.n-1):
            for j in range(1, self.m - 1):
                if (((i+j) % 2) == oddity):
                    num = (self.p[i+1,j]*self.K[i+1,j] + self.p[i-1,j]*self.K[i,j])/(self.hx * self.hx) + \
                                  (self.p[i,j+1]*self.K[i, j+1] + self.p[i,j-1]*self.K[i, j])/(self.hz * self.hz)
                    denom = (self.K[i+1, j] + self.K[i, j])/(self.hx * self.hx) + (self.K[i, j+1] + self.K[i, j])/(self.hz * self.hz) 
                    self.p[i,j] = num / denom

    def calc_div(self):
        self.div = np.zeros_like(self.p)
        for i in range(1, self.n-1):
            for j in range(1, self.m - 1):
              vxx = (self.K[i+1,j]*(self.p[i+1,j]-self.p[i,j]) - self.K[i,j]*(self.p[i,j]-self.p[i-1,j]) )/(self.hx * self.hx)
              vzz = (self.K[i,j+1]*(self.p[i,j+1]-self.p[i,j]) - self.K[i,j]*(self.p[i,j]-self.p[i,j-1]) )/(self.hz * self.hz)
              self.div[i, j] = vzz + vxx
        self.max_div = np.max(self.div)
