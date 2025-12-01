import numpy as np
from scipy.optimize import minimize


class GaussianProcess:
    def __init__ (self, X_train, y_train, kernel, sigma_n=1e-2):
        """
        Gaussian Process Regression Model.
        
        Args:
        X_train : (n1, d) - training inputs
        y_train : (n1, 1) - training targets
        kernel : callable - kernel function
        sigma_n : float - noise standard deviation
        """
        self.X_train = X_train
        self.y_train = y_train
        self.kernel = kernel
        self.sigma_n = sigma_n

        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K)  # (n1, n1) Cholesky decomposition K = L @ L.T

    def update_training_data(self, X_update, y_update):
        """
        Update training data with new points.
        
        Args:
            X_update : (m, d) - new training inputs
            y_update : (m, 1) - new training targets
        """
        self.X_train = np.append(self.X_train, X_update, 0)
        self.y_train = np.append(self.y_train, y_update, 0)
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K)  # (n1, n1) Cholesky decomposition K = L @ L.T

    def gp_predict(self, X_s):
        """
        Gaussian Process Prediction.

        Args:
            X_s : (n2, d) - test points
        
        Returns:
            mu : (n2, 1) - predicted mean
            cov : (n2, n2) - predicted covariance
        """
        K_s = self.kernel(self.X_train, X_s)  # (n1, n2)
        K_ss = self.kernel(X_s, X_s)  # (n2, n2)

        mu = K_s.T @ np.linalg.inv(self.K) @ self.y_train  # (n2, n1) @ (n1, n1) @ (n1, 1) = (n2, 1)
        cov = K_ss - K_s.T @ np.linalg.inv(self.K) @ K_s  # (n2, n2) - (n2, n1) @ (n1, n1) @ (n1, n2) = (n2, n2)

        return mu, cov

    def update_L(self, X_update, y_update):
        """
        Update Cholesky factor L with new training data.

        Args:
            X_update : (1, d) - new training input
            y_update : (1, 1) - new training target
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
        Stable Gaussian Process Prediction using Cholesky decomposition.
        
        Args:
            X_s : (n2, d) - test points
        
        Returns:
            mu : (n2, 1) - predicted mean
            cov : (n2, n2) - predicted covariance
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
        Optimize hyperparameters (length_scale, sigma_f, sigma_n) by maximizing the log marginal likelihood.

        Returns:
            opt_l : float - optimized length scale
            opt_sf : float - optimized signal variance
            opt_sn : float - optimized noise standard deviation
        """
        # Extract initial values from kernel object
        init_l = self.kernel.l
        init_sf = self.kernel.sf
        init_sn = self.sigma_n

        def negative_log_likelihood(params):
            """
            Compute the negative log marginal likelihood for given hyperparameters.
            
            Args:
                params : list - [length_scale, sigma_f, sigma_n]
                
            Returns:
                nll : float - negative log marginal likelihood
            """
            l, sf, sn = params

            # Update kernel with new parameters
            self.kernel.l = l
            self.kernel.sf = sf

            K = self.kernel(self.X_train, self.X_train) + sn**2 * np.eye(len(self.X_train))  # (n1, n1)

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
        bounds = [(1e-5, 10), (1e-5, 10), (1e-5, 10)]  # Ensure positive values
        res = minimize(negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')

        opt_l, opt_sf, opt_sn = res.x

        # Update kernel and noise with optimized parameters
        self.kernel.l = opt_l
        self.kernel.sf = opt_sf
        self.sigma_n = opt_sn

        # Recompute K and L with optimized parameters
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * np.identity(len(self.X_train))  # (n1, n1)
        self.L = np.linalg.cholesky(self.K)  # (n1, n1) Cholesky decomposition K = L @ L.T

        print(f"Optimized length_scale: {opt_l}, sigma_f: {opt_sf}, sigma_n: {opt_sn}")

        return opt_l, opt_sf, opt_sn
