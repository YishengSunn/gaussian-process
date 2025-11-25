import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel
from gp.utils import plot_gp


"""
# One time training and prediction example

# Training data
X_train = np.arange(-4, 4, 1).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.normal(0, 1e-2, (X_train.shape[0], 1))

# Create Gaussian Process model
kernel = RBF_kernel()
gp = GaussianProcess(X_train, y_train, kernel)

# Test data
X_s = np.linspace(-5, 5, 100).reshape(-1, 1)
Y_s = np.sin(X_s)

# Predict
mu_s, cov_s = gp.stable_gp_predict(X_s)

# Plot
plot_gp(gp, X_s, Y_s, mu_s, cov_s)
plt.show()
"""


# Iterate through training data points example

# Training data
X_train = np.arange(-4, 4, 1).reshape(-1, 1)
n = X_train.shape[0]
y_train = np.sin(X_train) + np.random.normal(0, 1e-2, (X_train.shape[0], 1))

# Test data
X_s = np.linspace(-5, 5, 100).reshape(-1, 1)
Y_s = np.sin(X_s)

# Create Gaussian Process model and different kernel
gp = None
kernel = RBF_kernel()
# kernel = Matern_kernel(nu=2.5)
# kernel = RationalQuadratic_kernel(alpha=1.0)
# kernel = Periodic_kernel(w=2*np.pi)

# Predict and plot iteratively
for i in range(n):
    print(f"Iteration {i}")
    if not gp:
        gp = GaussianProcess(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1), kernel)
    else:
        gp.update_L(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1))

    if i >= 5:
        gp.optimize_hyperparameters()
    mu_s, cov_s = gp.stable_gp_predict(X_s)

    plot_gp(gp, X_s, Y_s, mu_s, cov_s)

plt.show()
