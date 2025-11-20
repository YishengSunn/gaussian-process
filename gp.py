import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class RBF_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.l = length_scale
        self.sf = sigma_f

    def __call__(self, X1, X2):
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
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return (self.sf**2) * np.exp(-0.5 / self.l**2 * sqdist)  # (n1, n2)


class Matern_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, nu=1.5):
        self.l = length_scale
        self.sf = sigma_f
        self.nu = nu

    def __call__(self, X1, X2):
        """
        Matern kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the pairwise Euclidean distance between X1_i and X2_j
        pairwise_dists = np.linalg.norm((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2) / self.l  # (n1, n2)

        if self.nu == 0.5:
            K = self.sf**2 * np.exp(-pairwise_dists)
        elif self.nu == 1.5:
            sqrt3_dists = np.sqrt(3) * pairwise_dists
            K = self.sf**2 * (1 + sqrt3_dists) * np.exp(-sqrt3_dists)
        elif self.nu == 2.5:
            sqrt5_dists = np.sqrt(5) * pairwise_dists
            K = self.sf**2 * (1 + sqrt5_dists + (5/3) * pairwise_dists**2) * np.exp(-sqrt5_dists)

        return K


class RationalQuadratic_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, alpha=1.0):
        self.l = length_scale
        self.sf = sigma_f
        self.alpha = alpha

    def __call__(self, X1, X2):
        """
        Rational Quadratic kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the squared distance between X1_i and X2_j
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return self.sf**2 * (1 + sqdist / (2 * self.alpha * self.l**2)) ** (-self.alpha)  # (n1, n2)


class Periodic_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, w=1.0):
        self.l = length_scale
        self.sf = sigma_f
        self.w = w

    def __call__(self, X1, X2):
        """
        Periodic kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the pairwise Euclidean distance between X1_i and X2_j
        pairwise_dists = np.linalg.norm((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)  # (n1, n2)
        return self.sf**2 * np.exp(-1 * (np.sin(np.pi * pairwise_dists / self.w)**2) / (self.l**2))  # (n1, n2)
        

class GaussianProcess:
    def __init__ (self, X_train, y_train, kernel, sigma_n=1e-2, jitter=1e-8):
        """
        Gaussian Process Regression Model
        X_train : (n1, d) - training points
        y_train : (n1, 1) - training targets
        kernel : kernel function
        sigma_n : noise standard deviation of training data
        jitter : small value added to the diagonal for numerical stability
        """
        self.X_train = X_train
        self.y_train = y_train
        self.kernel = kernel
        self.sigma_n = sigma_n
        self.jitter = jitter

        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K + self.jitter*np.eye(self.K.shape[0]))  # (n1, n1) Cholesky decomposition K = L @ L.T

    def update_training_data(self, X_update, y_update):
        """
        Update training data with new point and recompute K and L
        X_update : (1, d) - new training point
        y_update : (1, 1) - new training target
        -------------
        """
        self.X_train = np.append(self.X_train, X_update, 0)
        self.y_train = np.append(self.y_train, y_update, 0)
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K + self.jitter*np.eye(self.K.shape[0]))  # (n1, n1) Cholesky decomposition K = L @ L.T

    def gp_predict(self, X_s):
        """
        Gaussian Process Prediction
        X_s : (n2, d) - test points
        -------------
        returns:
        f : (n2, 1) - predicted mean
        cov : (n2, n2) - predicted covariance
        -------------
        """

        K_s = self.kernel(self.X_train, X_s)  # (n1, n2)
        K_ss = self.kernel(X_s, X_s)  # (n2, n2)

        mu = K_s.T @ np.linalg.inv(self.K) @ self.y_train  # (n2, n1) @ (n1, n1) @ (n1, 1) = (n2, 1)
        cov = K_ss - K_s.T @ np.linalg.inv(self.K) @ K_s  # (n2, n2) - (n2, n1) @ (n1, n1) @ (n1, n2) = (n2, n2)

        return mu, cov

    def update_L(self, X_update, y_update):
        """
        Update the Cholesky decomposition L when new training data is added
        X_update : (1, d) - new training point
        y_update : (1, 1) - new training target
        -------------
        """
        self.X_train = np.append(self.X_train, X_update, 0)
        self.y_train = np.append(self.y_train, y_update, 0)

        k_star = self.kernel(self.X_train[:-1], X_update)  # (n, 1)
        k_star_star = self.kernel(X_update, X_update) + self.sigma_n**2  # (1, 1)

        # Compute the new row and column for L
        v = np.linalg.solve(self.L, k_star)  # (n, 1)
        L_right_bottom = np.sqrt(k_star_star - v.T @ v)  # (1, 1)

        # Update L
        n = self.L.shape[0]
        L_new = np.zeros((n+1, n+1))
        L_new[:n, :n] = self.L
        L_new[n, :n] = v.T
        L_new[n, n] = L_right_bottom.item()

        self.L = L_new
    
    def stable_gp_predict(self, X_s):
        """
        Stable Gaussian Process Prediction using Cholesky decomposition
        X_s : (n2, d) - test points
        -------------
        returns:
        f : (n2, 1) - predicted mean
        cov : (n2, n2) - predicted covariance
        -------------
        """
        
        K_s = self.kernel(self.X_train, X_s)  # (n1, n2)
        K_ss = self.kernel(X_s, X_s)  # (n2, n2)

        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))  # (n1, 1)
        v = np.linalg.solve(self.L, K_s)  # (n1, n2)
        
        mu = K_s.T @ alpha  # (n2, n1)
        cov = K_ss - v.T @ v
        
        return mu, cov
    
    def optimize_hyperparameters(self):
        """
        Optimize hyperparameters of the kernel and noise using Maximum Likelihood Estimation
        -------------
        """
        
        # Extract initial values from kernel object
        init_l = self.kernel.l
        init_sf = self.kernel.sf
        init_sn = self.sigma_n

        def negative_log_likelihood(params):
            """
            Computes -log p(y | X, θ) for optimizer.
            params : (3,) - [length_scale, sigma_f, sigma_n]
            -------------
            returns:
            nll : float - negative log likelihood
            -------------
            """

            l, sf, sn = params

            # Update kernel with new parameters
            self.kernel.l = l
            self.kernel.sf = sf

            K = self.kernel(self.X_train, self.X_train) + sn**2 * np.eye(len(self.X_train))  # (n1, n1)
            K += self.jitter * np.eye(K.shape[0])  # Add jitter for numerical stability

            try:
                L = np.linalg.cholesky(K)  # (n1, n1)
            except np.linalg.LinAlgError:
                return np.inf  # Return a large value if K is not positive definite
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train)) # (n1, 1)

            # log |K| = 2 * sum(log(diag(L)))
            log_det_K = 2 * np.sum(np.log(np.diag(L)))

            n = len(self.X_train)
            log_likelihood = -0.5 * self.y_train.T @ alpha - 0.5 * log_det_K - n / 2 * np.log(2 * np.pi)  # (1, 1)

            return -log_likelihood.squeeze()
        
        # Run the optimizer
        initial_params = [init_l, init_sf, init_sn]
        bounds = [(1e-5, None), (1e-5, None), (1e-5, None)]  # Ensure positive values
        res = minimize(negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')

        opt_l, opt_sf, opt_sn = res.x

        # Update kernel and noise with optimized parameters
        self.kernel.l = opt_l
        self.kernel.sf = opt_sf
        self.sigma_n = opt_sn

        # Recompute K and L with optimized parameters
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K + self.jitter*np.eye(self.K.shape[0]))  # (n1, n1) Cholesky decomposition K = L @ L.T

        print(f"Optimized length_scale: {opt_l}, sigma_f: {opt_sf}, sigma_n: {opt_sn}")

        return opt_l, opt_sf, opt_sn
    
    def plot(self, X_s, Y_s, mu_s, cov_s):
        """
        Plot the results
        Y_s : (n2, 1) - true function values at test points
        mu_s : (n2, 1) - predicted mean
        cov_s : (n2, n2) - predicted covariance
        -------------
        """

        std_s = np.sqrt(np.diag(cov_s))

        plt.figure(figsize = (10, 6))
        plt.plot(self.X_train.ravel(), self.y_train.ravel(), 'ro', label="Training Points")
        plt.plot(X_s, Y_s, 'y-', label="Real Curve")

        plt.plot(X_s, mu_s, 'b-', label="Predicted Curve")
        plt.fill_between(X_s.ravel(), mu_s.ravel()-3*std_s, mu_s.ravel()+3*std_s, alpha=0.2, label="Confidence Level (3σ)")

        plt.title("Gaussian Process Regression")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()


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
gp.plot(X_s, Y_s, mu_s, cov_s)
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
for i in range(1, n):
    if not gp:
        gp = GaussianProcess(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1), kernel)
    else:
        gp.update_L(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1))

    gp.optimize_hyperparameters()
    mu_s, cov_s = gp.stable_gp_predict(X_s)

    gp.plot(X_s, Y_s, mu_s, cov_s)

plt.show()
