import os
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_model import GaussianProcess
from gp.kernels import RBF_kernel, Matern_kernel, RationalQuadratic_kernel, Periodic_kernel
from utils.utils import rotate_to_fixed_frame, estimate_base_angle, rot2, wrap_pi, find_anchor_at_angle


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
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.train_line, = self.ax.plot([], [], 'r-', alpha=0.3)
        self.train_line.set_data(np.array(self.train_data)[:, 0], np.array(self.train_data)[:, 1]) if self.train_data is not None else None
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
            X (np.ndarray): input features of shape (M, window * 6).
            Y (np.ndarray): target outputs of shape (M, 3).
        """
        pos = np.asarray(positions)

        # Compute global base direction (average direction of first window segments)
        global_base_dir = estimate_base_angle(pos, n_segments=10)
        # phi0 = np.arctan2(global_base_dir[1], global_base_dir[0])

        origin = pos[0].copy()
        pos_t = pos - origin

        N = pos_t.shape[0]
        M = N - self.window
        
        r = np.linalg.norm(pos_t, axis=1)
        # theta = np.arctan2(pos_t[:, 1], pos_t[:, 0]) - phi0
        theta = np.arctan2(pos_t[:, 1], pos_t[:, 0])
        d_polar = np.column_stack((r, np.cos(theta), np.sin(theta)))

        # Inputs are absolute polar coordinates over the window
        X = np.zeros((M, self.window * 3), dtype=np.float64)
        Y = np.zeros((M, 2), dtype=np.float64)

        for i in range(M):
            X[i] = d_polar[i:i + self.window].flatten()
            Y[i] = rotate_to_fixed_frame(pos[i + self.window] - pos[i + self.window - 1], global_base_dir)

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

        self.gp_x = GaussianProcess(X, Y[:, [0]], kernel=self.kernel, sigma_n=self.sigma_n)
        self.gp_y = GaussianProcess(X, Y[:, [1]], kernel=self.kernel, sigma_n=self.sigma_n)

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
        Predict future positions using geometric invariant GP with explicit probe→reference alignment at a fixed anchor angle.
        
        Returns:
            np.ndarray: predicted future positions of shape (predict_steps + 1, 2).
        """
        if len(self.positions) < self.window + 2:
            return []
                
        # Reference data
        ref = np.asarray(self.train_data)
        ref_origin = ref[0]
        global_base_dir = estimate_base_angle(np.asarray(ref), n_segments=10)

        i_ref = find_anchor_at_angle(ref)
        if i_ref is None:
            return []
        
        ref_anchor = ref[i_ref]
        v_ref = ref_anchor - ref_origin
        nr = np.linalg.norm(v_ref)
        if nr < 1e-9:
            return []
        
        ang_ref = np.arctan2(v_ref[1], v_ref[0])

        # Probe data
        probe = np.asarray(self.positions)
        probe_origin = probe[0]

        i_probe = find_anchor_at_angle(probe)
        if i_probe is None:
            return []

        probe_anchor = probe[i_probe]
        v_probe = probe_anchor - probe_origin
        nn = np.linalg.norm(v_probe)
        if nn < 1e-9:
            return []

        ang_probe = np.arctan2(v_probe[1], v_probe[0])

        # Similarity transform
        dtheta = wrap_pi(ang_probe - ang_ref)
        scale = nn / nr

        # Probe -> Reference frame
        R_inv = rot2(-dtheta)
        probe_in_ref = (probe - probe_origin) @ R_inv.T / scale + ref_origin

        # GP rollout in reference frame
        pos = probe_in_ref.copy()

        for _ in range(self.predict_steps):
            pos_t = pos - ref_origin

            r = np.linalg.norm(pos_t, axis=1)
            theta = np.arctan2(pos_t[:, 1], pos_t[:, 0])
            d_polar = np.column_stack((r, np.cos(theta), np.sin(theta)))

            X_in = d_polar[-self.window:].flatten().reshape(1, -1)

            mu_dx, _ = self.gp_x.stable_gp_predict(X_in)
            mu_dy, _ = self.gp_y.stable_gp_predict(X_in)
            step_fixed = np.array([mu_dx[0, 0], mu_dy[0, 0]])

            # Map step back to world frame
            gb = global_base_dir / np.linalg.norm(global_base_dir)
            R = np.column_stack([gb, np.array([-gb[1], gb[0]])])
            step_world = step_fixed @ R.T
            new_p = pos[-1] + step_world

            pos = np.vstack((pos, new_p))

        preds_ref = pos[self.window - 1:]

        # Map back to probe frame
        R = rot2(dtheta)
        preds_probe = scale * ((preds_ref - ref_origin) @ R.T) + probe_origin

        return preds_probe

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
            self.fig.canvas.draw_idle()

            if self.online:
                self.online_train_gp()

    def on_release(self, event):
        """
        Handle mouse release events.
        """
        if event.button == 1:
            self.update_prediction_plot()


# Training circle
num_points = 500
R = 2.0
theta = np.linspace(np.pi/2, -3*np.pi/2, num_points)
training_circle = np.vstack((R * np.cos(theta), R * np.sin(theta))).T

# Test circle
num_points = 500
R = 3
center = np.array([1.2, -0.8])
theta = np.linspace(np.pi/2, -3*np.pi/2, num_points)
test_circle = center + np.vstack((R * np.cos(theta), R * np.sin(theta))).T

# Manual circle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "..", "data", "circle.npy")
manual_circle = np.array(np.load(path, allow_pickle=True)[0], dtype=float)

predictor = LiveCirclePredictor(window=10, predict_steps=150, online=False, train_data=training_circle[::2],
                                geo_gp=True, kernel_type='RBF', length_scale=5.0, sigma_f=1.0, sigma_n=1e-2)

test_start = 80
test_len = 180
assert test_len >= predictor.window + 1, "test_len should be > predictor.window"

predictor.positions = [list(p) for p in test_circle[test_start:test_start + test_len:2]]
xs = np.array(predictor.positions)[:, 0]
ys = np.array(predictor.positions)[:, 1]
predictor.drawn_line.set_data(xs, ys)
predictor.update_prediction_plot()

plt.show()
