
import numpy as np
import copy, math

def compute_cost(X, y, w, b):
  """
  args:
  X (ndarray): matrix of input training values.
  y (ndarray): 1-D vector of output training values.
  w (ndarray): 1-D vector of input weights.
  b (scalar): bias.

  output: total_cost (scalar)
  """
  m = X.shape[0]
  cost = 0.0
  for i in range (m):
    f_wb = np.dot(w, X[i]) + b
    cost = cost + (f_wb - y[i])**2
  cost = cost / (2*m)
  return cost

X = np.array([[1.0, 2.7, 3.2, 4.5], [3.1, 9.0, 3.0, 1.0], [0.8, 4.2, 6.8, 2.3]])
y = np.array([3.0, 6.12, 4.10])
w = np.array([0.2, 1.2, 3.0, 2.9])

def compute_gradient(X, y, w, b):
  """
  args:
  X (ndarray): matrix of input training values.
  y (ndarray): 1-D vector of output training values.
  w (ndarray): 1-D vector of input weights.
  b (scalar): bias.

  ouput: dj_dw, dj_db.
  """

  m,n = X.shape
  dj_dw = np.zeros([n,])
  dj_db = 0.0

  for i in range(m):
    err = (np.dot(w, X[i]) + b) - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err * X[i, j]
    dj_db = dj_db + err

  dj_dw = (1/m) * dj_dw
  dj_db = (1/m) * dj_db
  return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, num_iters, cost_function, gradient_function):
  """
  args:
  X (ndarray): matrix of input training values.
  y (ndarray): 1-D vector of output training values.
  w (ndarray): 1-D vector of input weights.
  b (scalar): bias.
  alpha: learning rate, choose small (e.g. 0.1, 0.03) value.
  num_iters: amount of iterations the gradient descent function will go through.
  cost_function: your implemented function to calculate the cost at any given moment.

  output:
  w (ndarray): optimized w parameter for multiple linear regression.
  b (scalar): optimized b parameter.
  J_history: evolution of cost function values through the iterations.
  """

  J_history = []
  w = copy.deepcopy(w_init)
  b = b_init

  for i in range(num_iters):
    dj_dw, dj_db = gradient_function(X, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    if i < 10000:
      J_history.append(cost_function(X, y, w, b))

  return w, b, J_history

print(gradient_descent(X, y, w, 0, 0.1, 10000, compute_cost, compute_gradient))

