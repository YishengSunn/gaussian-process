import os
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel
from gp.utils import animate_gp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "..", "outputs", "videos", "gp_animation.mp4")


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

animate_gp(X_train, y_train, X_s, Y_s, kernel, n, save=False, path=path)
