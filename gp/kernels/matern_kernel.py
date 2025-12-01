import numpy as np

class Matern_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, nu=1.5):
        """
        Matern kernel initialization.

        Args:
            length_scale (float): Length scale parameter.
            sigma_f (float): Signal variance.
            nu (float): Smoothness parameter. Common values are 0.5, 1.5, and 2.5.
        """
        self.l = length_scale
        self.sf = sigma_f
        self.nu = nu

    def __call__(self, X1, X2):
        """
        Compute the Matern kernel between two sets of input points.

        Args:
            X1 (np.ndarray): First set of input points of shape (n1, d).
            X2 (np.ndarray): Second set of input points of shape (n2, d).
        
        Returns:
            np.ndarray: Kernel matrix of shape (n1, n2).
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
