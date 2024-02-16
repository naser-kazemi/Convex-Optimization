# Let's address the tasks one by one with Python code.

# First, we need to import the necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Task 1: Data Preparation
df = pd.read_csv("age_salary.csv")
df = df.drop('User ID', axis=1)
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

X = df[['Age', 'EstimatedSalary']]
y = 2 * df['Purchased'] - 1  # Convert {0, 1} to {-1, 1}

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the data before training the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 2: SVM Training
# Train a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train_scaled, y_train)

# Task 3: Visualization
# Visualize the decision boundary of the trained SVM model
plt.figure(figsize=(10, 6))

# Plot the decision boundary
ax = plt.gca()
# Get the separating hyperplane
w = clf.coef_[0]
b = clf.intercept_[0]

# At the decision boundary, w0*x0 + w1*x1 + b = 0
# => x1 = - (w0*x0 + b) / w1
xx = np.linspace(X_train_scaled[:, 0].min(),
                 X_train_scaled[:, 0].max())
yy = - (w[0] * xx + b) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + w[0] ** 2) * margin
yy_up = yy + np.sqrt(1 + w[0] ** 2) * margin

ax.plot(xx, yy, 'k-')
ax.plot(xx, yy_down, 'k--')
ax.plot(xx, yy_up, 'k--')


# Plot data points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:,
            1], c=y_train, cmap=plt.cm.coolwarm)

plt.xlabel('Age (scaled)')
plt.ylabel('Salary (scaled)')
plt.title('SVM Decision Boundary with Support Vectors')
plt.show()


# Task 4: Model Evaluation
# Evaluate the model on the testing set
y_pred = clf.predict(X_test_scaled)
print("Classification report for SVM:")
print(classification_report(y_test, y_pred))

# Task 5: Reset and Repeat with make_circles
# Generate a two-dimensional dataset with a circular decision boundary
X_circle, y_circle = datasets.make_circles(n_samples=400, factor=.3, noise=.05)
y_circle = 2 * y_circle - 1  # Convert {0, 1} to {-1, 1}

# Split the dataset into a training set and a testing set
X_train_circle, X_test_circle, y_train_circle, y_test_circle = train_test_split(
    X_circle, y_circle, test_size=0.2, random_state=42)

# Scale the data
scaler_circle = StandardScaler()
X_train_circle_scaled = scaler_circle.fit_transform(X_train_circle)
X_test_circle_scaled = scaler_circle.transform(X_test_circle)

# Train the SVM classifier with linear kernel
clf_circle = svm.SVC(kernel='linear')
clf_circle.fit(X_train_circle_scaled, y_train_circle)

# Evaluate the classifier
y_pred_circle = clf_circle.predict(X_test_circle_scaled)
print("Classification report for SVM with linear kernel on circular data:")
print(classification_report(y_test_circle, y_pred_circle))

# Plot the decision boundary
plt.figure(figsize=(10, 6))
ax = plt.gca()
# Get the separating hyperplane
w = clf.coef_[0]
b = clf.intercept_[0]

# At the decision boundary, w0*x0 + w1*x1 + b = 0
# => x1 = - (w0*x0 + b) / w1
xx = np.linspace(X_train_circle_scaled[:, 0].min(),
                 X_train_circle_scaled[:, 0].max())
yy = - (w[0] * xx + b) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + w[0] ** 2) * margin
yy_up = yy + np.sqrt(1 + w[0] ** 2) * margin

ax.plot(xx, yy, 'k-')
ax.plot(xx, yy_down, 'k--')
ax.plot(xx, yy_up, 'k--')

plt.scatter(X_train_circle_scaled[:, 0], X_train_circle_scaled[:,
            1], c=y_train_circle, cmap=plt.cm.coolwarm)

plt.xlabel('x')
plt.ylabel('y')
plt.title(
    'SVM Decision Boundary with Support Vectors on Circular Data with Linear Kernel')
plt.show()


# Is linear kernel the best option here?
# No, for circular data, a linear kernel is not the best option as it cannot capture the circular decision boundary.
# A non-linear kernel like RBF would be more appropriate.
# Let's train an SVM with RBF kernel and evaluate it
clf_circle_rbf = svm.SVC(kernel='rbf')
clf_circle_rbf.fit(X_train_circle_scaled, y_train_circle)
y_pred_circle_rbf = clf_circle_rbf.predict(X_test_circle_scaled)
print("Classification report for SVM with RBF kernel on circular data:")
print(classification_report(y_test_circle, y_pred_circle_rbf))

# Plot the decision boundary of the RBF kernel SVM
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X_train_circle_scaled[:, 0].min(
) - 1, X_train_circle_scaled[:, 0].max() + 1
y_min, y_max = X_train_circle_scaled[:, 1].min(
) - 1, X_train_circle_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the function value for the whole grid
Z = clf_circle_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
plt.scatter(X_train_circle_scaled[:, 0], X_train_circle_scaled[:,
            1], c=y_train_circle, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title(
    'SVM Decision Boundary with Support Vectors on Circular Data with RBF Kernel')
plt.show()
