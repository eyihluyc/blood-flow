from constants import *

# finite difference grid:

# p_l and p_r -- pressure on borders, a function of z
class Grid():
    def __init__(self, p_l, p_r, n=500, m=100):
        self.n = n
        self.m = m
        
        self.hx = L/n
        self.hz = D/m

        self.p = np.zeros((n, m))
        self.p[0] = np.vectorize(p_l)(np.linspace(0, D, m))
        self.p[-1] = np.vectorize(p_r)(np.linspace(0, D, m))
        
        self.K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                self.K[i,j] = K(i*self.hx, j*self.hz)
        
    # initial values -- linear interpolation b/w left and right boundaries
        for j in range(0, m):
            self.p[:, j] = np.interp(np.arange(0, n), [0, n-1], [self.p[0, j], self.p[n-1, j]])
        
        self.iterations = 0
        self.eps = float('inf')
        
    # apply gauss-seidel method with even/odd order until distance with L\inf is less than eps
    def iterate(self, iter_or_err="iter", steps=100, eps=1e-5):
        
        if (iter_or_err == "iter"):
            for _ in range(steps-1):
                self.make_step()
            self.calculate_eps()
        elif (iter_or_err == "err"):
            while (self.eps > eps):
                self.calculate_eps()
    
    # method invokes one step
    def calculate_eps(self):
        prev = self.p.copy()
        self.make_step()
        self.eps = np.max(np.abs(prev - self.p))
        return self.eps
        
    
    def make_step(self):
        self.iterations += 1
        self.half_step(0)
        self.half_step(1)
        
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]
    
    def half_step(self, oddity):

        for i in range(1, self.n-1):
            for j in range(1, self.m - 1):
                if (((i+j) % 2) == oddity):
                    num = (self.p[i+1,j]*self.K[i+1,j] + self.p[i-1,j]*self.K[i,j])/(self.hx * self.hx) + \
                                  (self.p[i,j+1]*self.K[i, j+1] + self.p[i,j-1]*self.K[i, j])/(self.hz * self.hz)
                    denom = (self.K[i+1, j] + self.K[i, j])/(self.hx * self.hx) + (self.K[i, j+1] + self.K[i, j])/(self.hz * self.hz) 
                    self.p[i,j] = num / denom