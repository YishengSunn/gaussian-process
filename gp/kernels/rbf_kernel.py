import numpy as np

class RBF_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        """
        Radial Basis Function kernel.

        Args:
            length_scale: Length scale parameter (l)
            sigma_f: Signal variance (σ_f)
        """
        self.l = length_scale
        self.sf = sigma_f

    def __call__(self, X1, X2):
        """
        Compute the RBF kernel between two sets of input points.

        Args:
            X1: First set of input points, shape (n1, d)
            X2: Second set of input points, shape (n2, d)

        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Calculate the squared distance between X1_i and X2_j
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return (self.sf**2) * np.exp(-0.5 / self.l**2 * sqdist)  # (n1, n2)
