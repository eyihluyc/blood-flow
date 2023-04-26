import numpy as np
from tqdm import tqdm

from constants import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# esp with which derivatives of Psi are calculated
EPSILON = 1e-4

# 1 layer
class NN4PDE:
  def __init__(self, BC, hidden=10):
    self.BC = BC
    self.hidden = hidden
    # weights input-hidden, hidden bias, hidden-output, output bias
    # TODO!!! Input layer has dim 2
    self.params = np.random.normal(0, 1, 4 * self.hidden + 1)
    self.epochs = 0

  # feed-forward NN with input(2), 1 hidden(self.hidden) and output layers(1)
  def NN(self, x, z, params):
    # print("NN")
    p = x * params[0:self.hidden] + z * params[self.hidden:self.hidden*2] + params[self.hidden*2:self.hidden*3]
    p = np.vectorize(sigmoid)(p)
    p = np.dot(p, params[self.hidden*3:self.hidden*4]) + params[-1]
    p = np.tanh(p)
    return p
  
  ###########
  # Derivatives of the NN
  ###########
  def dNdz(self, x, z, params, eps=EPSILON):
    return (self.NN(x, z+eps, params) - self.NN(x-eps, z, params)) / (2*eps)

  ###########
  # Derivatives of the trial solution (Psi)
  ###########
  def Psi(self, x, z, params=None):
    # print("Psi")
    if params is None:
      params = self.params
    return self.BC(x, z) + x * (1-x) * (self.NN(x, z, params) - (z - z*z/2) * self.dNdz(x, 0, params) - z*z/2 * self.dNdz(x, 1, params))

  def dPsidx(self, x, z, params, eps=EPSILON):
    return (self.Psi(x+eps, z, params) - self.Psi(x-eps, z, params)) / (2*eps)
  
  def dPsidz(self, x, z, params, eps=EPSILON):
    return (self.Psi(x, z+eps, params) - self.Psi(x-eps, z, params)) / (2*eps)
  
  def d2Psidx2(self, x, z, params, eps=EPSILON):
    prev = self.Psi(x-eps, z, params)
    cur = self.Psi(x, z, params)
    next = self.Psi(x+eps, z, params)
    return (next + prev - 2*cur) / (eps * eps)
  
  def d2Psidz2(self, x, z, params, eps=EPSILON):
    prev = self.Psi(x, z-eps, params)
    cur = self.Psi(x, z, params)
    next = self.Psi(x, z+eps, params)
    return (next + prev - 2*cur) / (eps * eps)
  

  # How to use the network
  def eval(self, pts):
    # print("eval")
    def eval_pt(point):
      x, z = point
      return self.Psi(x, z, self.params)
    return np.vectorize(eval_pt)(pts)


  # Cost and gradient of a cost w.r.t. params
  def cost(self, pts, params=None):
    # print("cost")

    if params is None:
      params = self.params
    
    def cost1(x, z):
      # print("cost1")
      return (self.d2Psidx2(x, z, params) + self.d2Psidz2(x, z, params) + \
              dKdx(x, z) * self.dPsidx(x, z, params) + dKdz(x, z) * self.dPsidz(x, z, params)) **2

    return sum(map(lambda point: cost1(point[0], point[1]), pts)) / len(pts)

  def dCOSTdp_i(self, pts, i, eps=EPSILON):
    # print("dCOSTdp_i")

    eps_vectorized = np.zeros(len(self.params))
    eps_vectorized[i] = eps
    return (self.cost(pts, self.params + eps_vectorized) - self.cost(pts, self.params - eps_vectorized)) / (2*eps)

  def grad_cost(self, pts):
    # gradient of cost function w.r.t. params
    # print("grad_cost")

    grad_vector = [self.dCOSTdp_i(pts, i) for i in range(len(self.params))]
    return np.array(grad_vector)

  def train(self, input_pts, LR=0.05, MOMENTUM=0.95, EPOCHS=200):
    # print("train")

    velocities = np.zeros(len(self.params))
    for i in tqdm(range(EPOCHS)):
      grad = self.grad_cost(input_pts)
      velocities = MOMENTUM * velocities - LR * grad
      self.params += velocities
      
      if (i % 10 == 0):
        print(f"Epoch: {i}, cost: {self.cost(input_pts)}")
    print(f"Epoch: {i}, Final cost: {self.cost(input_pts)}")
    self.epochs = EPOCHS
