import os
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel


class LiveCirclePredictor:
    def __init__(self, window=10, predict_steps=20, online=True, train_data=None, 
                 geo_gp=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, 
                 sigma_n=1e-2):
        """
        Live predictor for circle drawing using Gaussian Processes.

        Args:
            window (int): number of previous points to consider.
            predict_steps (int): number of future points to predict.
            online (bool): whether to use online training.
            train_data (np.ndarray): training data for offline training.
            geo_gp (bool): whether to use geometric invariant GP.
            kernel_type (str): type of kernel to use ('RBF', 'Matern', 'RationalQuadratic', 'Periodic').
            length_scale (float): length scale parameter for the kernel.
            sigma_f (float): signal variance for the kernel.
            sigma_n (float): noise variance for GP.
        """
        self.window = window
        self.predict_steps = predict_steps
        self.online = online
        self.train_data = train_data
        self.geo_gp = geo_gp
        self.sigma_n = sigma_n

        # Initialize kernel
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

        # Initialize data storage
        self.positions = []

        # Initialize GP models
        self.gp_x = None
        self.gp_y = None
        self.gp_r = None
        self.gp_sin = None
        self.gp_cos = None

        if self.geo_gp:
            if not self.online and self.train_data is None:
                raise ValueError("train_data must be provided for offline training.")
            elif not self.online:
                self.offline_train_geometric_invariant_gp()
        else:
            if not self.online and self.train_data is None:
                raise ValueError("train_data must be provided for offline training.")
            elif not self.online:
                self.offline_train_gp()

        # Set up figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.set_title("Live Circle Drawing with GP Prediction")
        self.ax.grid(True, alpha=0.3)

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.drawn_line, = self.ax.plot([], [], 'b-')
        self.pred_line, = self.ax.plot([], [], 'g--', alpha=0.5)

    def build_autoregressive_dataset(self, positions):
        """
        Build autoregressive dataset from positions.

        Args:
            positions (list of (x, y)): list of recorded positions.

        Returns:
            X (np.ndarray): input features of shape (M, window * 2).
            Y (np.ndarray): target outputs of shape (M, 2).
        """
        pos = np.array(positions)
        N = pos.shape[0]
        M = N - self.window

        X = np.zeros((M, self.window * 2))
        Y = np.zeros((M, 2))

        for i in range(M):
            X[i] = pos[i:i + self.window].flatten()
            Y[i] = pos[i + self.window] - pos[i + self.window - 1]
        
        return X, Y
    
    def build_geometric_invariant_dataset(self, positions):
        """
        Build geometric invariant dataset from positions.

        Args:
            positions (list of (x, y)): list of recorded positions.

        Returns:
            X (np.ndarray): input features of shape (M, window * 3).
            Y (np.ndarray): target outputs of shape (M, 3).
        """
        pos = np.array(positions)
        pos = pos - pos[0]  # Translate to origin

        N = pos.shape[0]
        M = N - self.window
        
        r = np.linalg.norm(pos, axis=1)
        theta = np.arctan2(pos[:, 1], pos[:, 0])
        d_polar = np.column_stack((r, np.cos(theta), np.sin(theta)))

        # Normalize to [0, 1]
        # d_polar[:, 0] /= np.max(d_polar[:, 0])
        d_polar[:, 1] = (d_polar[:, 1] + 1) / 2
        d_polar[:, 2] = (d_polar[:, 2] + 1) / 2

        # Inputs are absolute and relative polar coordinates over the window
        X = np.zeros((M, self.window * 6))
        Y = np.zeros((M, 3))

        for i in range(M):
            X[i, :self.window * 3] = d_polar[i:i + self.window].flatten()
            if i > 0:
                X[i, self.window * 3:] = (d_polar[i:i + self.window] - d_polar[i - 1:i + self.window - 1]).flatten()
            else:
                X[i, self.window * 3:] = np.zeros(self.window * 3)
            Y[i] = d_polar[i + self.window] - d_polar[i + self.window - 1]

        return X, Y

    def online_train_gp(self):
        """
        Online training of GP with the latest position.
        """
        if len(self.positions) < self.window + 1:
            return None

        if not self.gp_x or not self.gp_y:
            X, Y = self.build_autoregressive_dataset(self.positions)

            self.gp_x = GaussianProcess(X, Y[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
            self.gp_y = GaussianProcess(X, Y[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)

        else:
            X_new = np.array(self.positions[-1 - self.window:-1]).flatten().reshape(1, -1)
            Y_new = (np.array(self.positions[-1]) - np.array(self.positions[-2])).reshape(1, -1)

            self.gp_x.update_L(X_new, Y_new[:, [0]])
            self.gp_y.update_L(X_new, Y_new[:, [1]])

    def offline_train_gp(self):
        """
        Offline training of GP using provided training data.
        """
        X, Y = self.build_autoregressive_dataset(self.train_data)

        self.gp_x = GaussianProcess(X, Y[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_y = GaussianProcess(X, Y[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)

    def offline_train_geometric_invariant_gp(self):
        """
        Offline training of GP using geometric invariant dataset.
        """
        X, Y = self.build_geometric_invariant_dataset(self.train_data)

        self.gp_r = GaussianProcess(X, Y[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_cos = GaussianProcess(X, Y[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_sin = GaussianProcess(X, Y[:, [2]], kernel=self.kernel, sigma_n=self.sigma_n)

    def predict_future(self):
        """
        Predict future positions based on current positions.

        Returns:
            np.ndarray: predicted future positions of shape (predict_steps + 1, 2).
        """
        if len(self.positions) < self.window + 1:
            return []

        prev = np.array(self.positions[-self.window:])

        for _ in range(self.predict_steps):
            X_in = prev[-self.window:].flatten().reshape(1, -1)

            mu_dx, _ = self.gp_x.stable_gp_predict(X_in)
            mu_dy, _ = self.gp_y.stable_gp_predict(X_in)

            new_p = prev[-1] + np.array([mu_dx[0, 0], mu_dy[0, 0]])
            prev = np.vstack((prev, new_p))

        return prev[self.window - 1:]
    
    def predict_geometric_invariant_future(self):
        """
        Predict future positions using geometric invariant GP.
        
        Returns:
            np.ndarray: predicted future positions of shape (predict_steps + 1, 2).
        """
        if len(self.positions) < self.window + 1:
            return []
        
        pos = np.array(self.positions)

        for _ in range(self.predict_steps):
            pos_translated = pos - pos[0]

            r = np.linalg.norm(pos_translated, axis=1)
            theta = np.arctan2(pos_translated[:, 1], pos_translated[:, 0])
            d_polar = np.column_stack((r, np.cos(theta), np.sin(theta)))

            # d_polar[:, 0] /= np.max(d_polar[:, 0])
            d_polar[:, 1] = (d_polar[:, 1] + 1) / 2
            d_polar[:, 2] = (d_polar[:, 2] + 1) / 2

            X_in = np.zeros((1, self.window * 6))
            X_in[0, :3 * self.window] = d_polar[-self.window:].flatten()
            X_in[0, 3 * self.window:] = (d_polar[-self.window:] - d_polar[-self.window - 1:-1]).flatten()

            mu_dr, _ = self.gp_r.stable_gp_predict(X_in)
            mu_dcos, _ = self.gp_cos.stable_gp_predict(X_in)
            mu_dsin, _ = self.gp_sin.stable_gp_predict(X_in)

            new_r = d_polar[-1, 0] + mu_dr[0, 0]
            new_cos = (d_polar[-1, 1] + mu_dcos[0, 0]) * 2 - 1
            new_sin = (d_polar[-1, 2] + mu_dsin[0, 0]) * 2 - 1
            new_theta = np.arctan2(new_sin, new_cos)

            new_x = new_r * np.cos(new_theta)
            new_y = new_r * np.sin(new_theta)

            new_p = pos[0] + np.array([new_x, new_y])
            pos = np.vstack((pos, new_p))
        
        return pos[self.window - 1:]

    def update_prediction_plot(self):
        """
        Update the prediction plot with future predictions.
        """
        preds = self.predict_geometric_invariant_future() if self.geo_gp else self.predict_future()
        if len(preds) > 0:
            self.pred_line.set_data(preds[:, 0], preds[:, 1])
            self.fig.canvas.draw_idle()

    def on_press(self, event):
        """
        Handle mouse press events.
        """
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Remove previous points and predictions
            self.positions = []
            self.drawn_line.set_data([], [])
            self.pred_line.set_data([], [])

            # Add the first point
            self.positions.append([event.xdata, event.ydata])
            self.drawn_line.set_data(np.array(self.positions)[:, 0], np.array(self.positions)[:, 1])
            self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """
        Handle mouse motion events.
        """
        if (len(self.positions) == 0):
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Add new point
            self.positions.append([event.xdata, event.ydata])
            self.drawn_line.set_data(np.array(self.positions)[:, 0], np.array(self.positions)[:, 1])

            if self.online:
                self.online_train_gp()
            self.update_prediction_plot()


# Perfect circle
num_points = 400
R = 2.0
theta = np.linspace(np.pi/2, -3*np.pi/2, num_points)
perfect_circle = np.vstack((R * np.cos(theta), R * np.sin(theta))).T

# # Manual circle
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path = os.path.join(BASE_DIR, "..", "data", "circle", "circle.npy")
# manual_circle = np.array(np.load(path, allow_pickle=True)[0], dtype=float)

# live_circle_predictor = LiveCirclePredictor(window=10, predict_steps=10, online=False, train_data=manual_circle[::2], 
#                                             geo_gp=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, sigma_n=1e-2)
# plt.show()


train_data_for_model = perfect_circle

predictor = LiveCirclePredictor(
    window=10,
    predict_steps=200,
    online=False,
    train_data=train_data_for_model,
    geo_gp=True,
    kernel_type='RBF',
    length_scale=5.0,
    sigma_f=1.0,
    sigma_n=1e-2
)

seed_len = 80
assert seed_len >= predictor.window, "seed_len should be >= predictor.window"

predictor.positions = [list(p) for p in perfect_circle[100:100+seed_len]]
xs = np.array(predictor.positions)[:, 0]
ys = np.array(predictor.positions)[:, 1]
predictor.drawn_line.set_data(xs, ys)
predictor.update_prediction_plot()

plt.show()
