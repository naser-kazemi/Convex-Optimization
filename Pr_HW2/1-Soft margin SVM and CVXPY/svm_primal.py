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


# weights(w): a vector variables of dimension d as variables
w = cp.Variable(d)

# bias(b): a scalar as variables
b = cp.Variable()

# slack variables(ksi): a set of n scalar variables as variables
ksi = cp.Variable(n)

# create regularization parameter(C): a scalar variable as parameter
C = cp.Parameter(nonneg=True)
C.value = 1

# objective function: 0.5 * ||w||^2 + C * sum(ksi)
objective = 0.5 * cp.sum_squares(w) + C * cp.sum(ksi)

# constraints:
constraints = []
for i in range(n):
    constraints.append(y[i] * (X[i] @ w + b) >= 1 - ksi[i])
    constraints.append(ksi[i] >= 0)


# problem definition
prob = cp.Problem(cp.Minimize(objective), constraints)

# problem solution
prob.solve()

# print solution
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var w", w.value)
print("optimal var b", b.value)

#  plot the points and the hyperplane, color the area according to the predicted class
xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
yy = - (w.value[0] * xx + b.value) / w.value[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
margin = 1 / np.sqrt(np.sum(w.value ** 2))
yy_down = yy - np.sqrt(1 + w.value[0] ** 2) * margin
yy_up = yy + np.sqrt(1 + w.value[0] ** 2) * margin

# Plot the points, the hyperplane, and the margins
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='winter', s=100)
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Shade the margin area
plt.fill_between(xx, yy_down, yy_up, alpha=0.1, color='k')

# Color the plane according to the distance from the hyperplane
xx_dense, yy_dense = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                                 np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
Z = np.c_[xx_dense.ravel(), yy_dense.ravel()] @ w.value + b.value
Z = Z.reshape(xx_dense.shape)

plt.contourf(xx_dense, yy_dense, Z, levels=np.linspace(
    Z.min(), Z.max(), 100), cmap='coolwarm', alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Margin')
plt.show()