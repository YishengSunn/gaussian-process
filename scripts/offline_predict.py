import os
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel


class CirclePredictor:
    def __init__(self, window=10, predict_delta=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, sigma_n=1e-2):
        """
        Initialize CirclePredictor with GP kernel and parameters.

        Args:
            window (int): Number of past positions to use for prediction.
            predict_delta (bool): Whether to predict position delta or absolute position.
            kernel_type (str): Type of kernel to use ('RBF', 'Matern', 'RationalQuadratic', 'Periodic').
            length_scale (float): Length scale parameter for the kernel.
            sigma_f (float): Signal variance for the kernel.
            sigma_n (float): Noise variance for the GP.
        """
        self.window = window
        self.sigma_n = sigma_n
        self.predict_delta = predict_delta

        if kernel_type == 'RBF':
            self.kernel = RBF_kernel(length_scale=length_scale, sigma_f=sigma_f)
        elif kernel_type == 'Matern':
            self.kernel = Matern_kernel(length_scale=length_scale, sigma_f=sigma_f)
        elif kernel_type == 'RationalQuadratic':
            self.kernel = RationalQuadratic_kernel(length_scale=length_scale, sigma_f=sigma_f)
        elif kernel_type == 'Periodic':
            self.kernel = Periodic_kernel(length_scale=length_scale, sigma_f=sigma_f)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        self.gp_x = None
        self.gp_y = None

    def build_autoregressive_dataset(self, positions):
        """
        Build autoregressive dataset from position data.

        Args:
            positions (np.ndarray): Array of shape (N, 2) with x and y positions.
        
        Returns:
            X (np.ndarray): Input features of shape (M, window*2).
            Y (np.ndarray): Target outputs of shape (M, 2).
        """
        N = positions.shape[0]
        M = N - self.window

        X = np.zeros((M, self.window*2))
        Y = np.zeros((M, 2))

        for i in range(M):
            past = positions[i : i + self.window]  # shape (window, 2)
            X[i, :] = past.flatten()  # order: x0, y0, x1, y1, ...

            target_idx = i + self.window
            if self.predict_delta:
                Y[i, :] = positions[target_idx] - positions[target_idx - 1]  # delta relative to previous sample
            else:
                Y[i, :] = positions[target_idx]
        
        return X, Y
    
    def train_gp(self, X_train, Y_train):
        """
        Train separate GPs for x and y coordinates.
        """
        self.gp_x = GaussianProcess(X_train, Y_train[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_y = GaussianProcess(X_train, Y_train[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)

    def recursive_gp_prediction(self, previous_positions, num_steps):
        """
        Perform recursive GP prediction for a number of steps.

        Args:
            previous_positions (np.ndarray): Array of shape (window, 2) with the last known positions.
            num_steps (int): Number of future steps to predict.
        
        Returns:
            predicted_positions (np.ndarray): Array of shape (num_steps+1, 2) with predicted positions.
            predicted_std (np.ndarray): Array of shape (num_steps+1, 2) with predicted standard deviations.
        """
        predicted_positions = [previous_positions[-1]]  # Start from the last known position
        predicted_std = [np.array([0.0, 0.0])]  # No uncertainty for the initial position

        for _ in range(num_steps):
            # Prepare input vector
            X_input = previous_positions.flatten().reshape(1, -1)  # shape (1, window*2)

            # Predict delta or absolute position
            mu_dx, cov_dx = self.gp_x.stable_gp_predict(X_input)
            mu_dy, cov_dy = self.gp_y.stable_gp_predict(X_input)

            if self.predict_delta:
                new_position = predicted_positions[-1] + np.array([mu_dx[0, 0], mu_dy[0, 0]])
            else:
                new_position = np.array([mu_dx[0, 0], mu_dy[0, 0]])

            # Update previous_positions for next prediction
            previous_positions = np.vstack((previous_positions[1:], new_position))

            # Store predictions
            predicted_positions.append(new_position)
            predicted_std.append(np.array([np.sqrt(cov_dx[0, 0]), np.sqrt(cov_dy[0, 0])]))

        return np.array(predicted_positions), np.array(predicted_std)
    
    def plot_prediction(self, train_positions, predicted_positions, predicted_std):
        """
        Plot training positions and GP predictions with confidence intervals.
        
        Args:
            train_positions (np.ndarray): Array of shape (N, 2) with training positions.
            predicted_positions (np.ndarray): Array of shape (M, 2) with predicted positions.
            predicted_std (np.ndarray): Array of shape (M, 2) with predicted standard deviations.
        """
        plt.figure(figsize=(6, 6))

        plt.plot(train_positions[:, 0], train_positions[:, 1], 'r-', label='Training Circle')
        plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'b--', label='GP Prediction')
        plt.fill_between(predicted_positions[:, 0],
                         predicted_positions[:, 1]-3*predicted_std[:, 1],
                         predicted_positions[:, 1]+3*predicted_std[:, 1],
                         alpha=0.2, label='Confidence Level (3σ)')

        plt.title('Gaussian Process Prediction of Circle')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)


# ----------------------------
# 1. Prepare Data
# ----------------------------
# Perfect circle
num_points = 400
R = 2.0
theta = np.linspace(np.pi/2, -3*np.pi/2, num_points)
perfect_circle = np.vstack((R * np.cos(theta), R * np.sin(theta))).T

# Manual circle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "..", "data", "circle", "strokes.npy")
manual_circle = np.array(np.load(path, allow_pickle=True)[0], dtype=float)

# Choose training & test dataset
train_on_manual, test_on_manual = False, True
train_data = manual_circle if train_on_manual else perfect_circle
test_data = manual_circle if test_on_manual else perfect_circle


# ----------------------------
# 2. Build dataset and train GP
# ----------------------------
# Initialize predictor
predictor = CirclePredictor(window=10, predict_delta=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, sigma_n=1e-2)

X, Y = predictor.build_autoregressive_dataset(train_data)

train_size = int(1.0 * X.shape[0])
predictor.train_gp(X[:train_size], Y[:train_size])


# ----------------------------
# 3. Prediction
# ----------------------------
test_start = 200

# Recursive prediction starting from the end of training data
initial_window = test_data[test_start : test_start + predictor.window]

pred_positions, pred_std = predictor.recursive_gp_prediction(
    previous_positions=initial_window,
    num_steps=test_data.shape[0] - predictor.window - test_start
)


# ----------------------------
# 4. Plot results
# ----------------------------
predictor.plot_prediction(
    train_positions=train_data[:train_size + predictor.window],
    predicted_positions=pred_positions,
    predicted_std=pred_std
)

if test_on_manual:
    plt.plot(test_data[:test_start + predictor.window, 0], test_data[:test_start + predictor.window, 1], 'c.', label='Manual Data Points')
else:
    plt.plot(test_data[:test_start + predictor.window, 0], test_data[:test_start + predictor.window, 1], 'c.', label='Perfect Circle Points')

plt.legend()
plt.show()
