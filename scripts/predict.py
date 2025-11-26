import os
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel


class CirclePredictor:
    def __init__(self, window=10, predict_delta=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, sigma_n=1e-2):
        """
        window: number of past positions used as input
        predict_delta: if True, predict delta relative to last input position
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
        Build autoregressive dataset from positions
        ------------------------------------------------
        Inputs: 
        positions: (N, 2) array of (x, y)
        ------------------------------------------------
        Returns: X (M, window*2), Y (M, 2)
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
        Train Gaussian Process models for x and y
        ------------------------------------------------
        Inputs: 
        X_train: (M, window*2) array of input features
        Y_train: (M, 2) array of target outputs
        ------------------------------------------------
        """
        self.gp_x = GaussianProcess(X_train, Y_train[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_y = GaussianProcess(X_train, Y_train[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)

    def recursive_gp_prediction(self, previous_positions, num_steps):
        """
        Perform recursive GP prediction starting from initial_position
        ---------------------------------------------------------------
        Inputs: 
        previous_positions: (window, 2) array of past positions
        num_steps: number of prediction steps
        ---------------------------------------------------------------
        Returns: 
        predicted_positions : (num_steps+1, 2) array of predicted positions
        predicted_std : (num_steps+1, 2) array of predicted standard deviations
        ---------------------------------------------------------------
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
        Plot the training positions, test positions, and GP predictions with uncertainty
        ---------------------------------------------------------------
        Inputs: 
        train_positions: (N_train, 2) array of training positions
        predicted_positions: (N_test+1, 2) array of predicted positions
        predicted_std: (N_test+1, 2) array of predicted standard deviations
        ---------------------------------------------------------------
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
