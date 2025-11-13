import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function kernel
    X1 : (n1, d)
    X2 : (n2, d)
    -------------
    returns:
    K : (n1, n2) - kernel matrix
    -------------
    """

    # Calculate the squared distance between X1_i and X2_j
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T) # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
    return (sigma_f**2) * np.exp(-0.5 / length_scale**2 * sqdist) # (n1, n2)


def gp_predict(X_train, Y_train, X_s, sigma_n=1e-2):
    """
    Gaussian Process Prediction
    X_train : (n1, d)
    Y_train : (n1, 1)
    X_s : (n2, d)
    sigma_n : noise standard deviation
    -------------
    returns:
    f : (n2, 1) - predicted mean
    cov : (n2, n2) - predicted covariance
    -------------
    """

    K = rbf_kernel(X_train, X_train) + sigma_n**2 * np.identity(np.shape(X_train)[0]) # (n1, n1)
    K_s = rbf_kernel(X_train, X_s) # (n1, n2)
    K_ss = rbf_kernel(X_s, X_s) # (n2, n2)

    mu = K_s.T @ np.linalg.inv(K) @ Y_train # (n2, n1) @ (n1, n1) @ (n1, 1) = (n2, 1)
    cov = K_ss - K_s.T @ np.linalg.inv(K) @ K_s # (n2, n2) - (n2, n1) @ (n1, n1) @ (n1, n2) = (n2, n2)

    return mu, cov


# Training data
X_train = np.arange(-4, 4, 1).reshape(-1, 1)
Y_train = np.sin(X_train) + np.random.normal(0, 1e-2, (X_train.shape[0], 1))

# Test data
X_s = np.linspace(-5, 5, 100).reshape(-1, 1)
Y_s = np.sin(X_s)

# Predict
mu_s, cov_s = gp_predict(X_train, Y_train, X_s)
std_s = np.sqrt(np.diag(cov_s))

# Plot
plt.figure(figsize = (10, 6))
plt.plot(X_train.ravel(), Y_train.ravel(), 'ro', label="Training Points")
plt.plot(X_s, Y_s, 'y-', label="Real Curve")
plt.plot(X_s, mu_s, 'b-', label="Predicted Curve")
plt.fill_between(X_s.ravel(), mu_s.ravel()-3*std_s, mu_s.ravel()+3*std_s, alpha=0.2, label="Confidence Level (3σ)")
plt.legend()
plt.show()