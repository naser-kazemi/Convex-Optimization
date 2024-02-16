import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp


# read the data, the first row is the label
data_path = 'svm_train.txt'
data = np.loadtxt(data_path, delimiter=' ', dtype=np.float64, skiprows=1)
X = data[:, 0:2]
y = data[:, 2]

n = X.shape[0]  # number of samples
d = X.shape[1]  # number of features


# alpha: Lagrange multipliers, a vector variables of dimension n as variables
alpha = cp.Variable(n)

# C: regularization parameter, a scalar variable as parameter
C = cp.Parameter(nonneg=True)
C.value = 1

# convert X to cp Matrix
X = cp.Constant(X)


# convert y to Y which is the diagonal matrix of y
y = cp.Constant(y)
Y = cp.diag(y)

P = cp.atoms.affine.wraps.psd_wrap(Y @ X @ X.T @ Y)

# objective function: sum(alpha) - 0.5 * alpha^T * Y * X * X^T * Y * \alpha
objective = cp.sum(alpha) - 0.5 * cp.quad_form(alpha, P)

# constraints:
constraints = []
for i in range(n):
    constraints.append(alpha[i] >= 0)
    constraints.append(alpha[i] <= C)

constraints.append(alpha @ y == 0)


# problem definition
prob = cp.Problem(cp.Maximize(objective), constraints)

# problem solution
prob.solve()

# print solution
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var alpha", alpha.value)


# now recover w and b from alpha
w = X.T @ Y @ alpha
b = cp.mean(y.value - X @ w.value)
print("optimal var w", w.value)
print("optimal var b", b.value)
