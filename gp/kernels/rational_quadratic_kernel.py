import numpy as np

class RationalQuadratic_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, alpha=1.0):
        """
        Rational Quadratic kernel.

        Args:
            length_scale : float : length scale parameter
            sigma_f : float : signal variance
            alpha : float : shape parameter
        """
        self.l = length_scale
        self.sf = sigma_f
        self.alpha = alpha

    def __call__(self, X1, X2):
        """
        Compute the Rational Quadratic kernel between two sets of input points.

        Args:
            X1 : np.ndarray : first set of input points, shape (n1, d)
            X2 : np.ndarray : second set of input points, shape (n2, d)

        Returns:
            np.ndarray : kernel matrix, shape (n1, n2)
        """
        # Calculate the squared distance between X1_i and X2_j
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return self.sf**2 * (1 + sqdist / (2 * self.alpha * self.l**2)) ** (-self.alpha)  # (n1, n2)
