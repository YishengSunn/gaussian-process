import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gp.gp_model import GaussianProcess


def plot_gp(gp, X_s, Y_s, mu_s, cov_s):
    """
    Plot the GP regression results.

    Args:
        gp: GaussianProcess object - trained GP model
        X_s: (n, 1) - test inputs
        Y_s: (n, 1) - true function values at test inputs
        mu_s: (n, 1) - predicted means at test inputs
        cov_s: (n, n) - predicted covariance matrix at test inputs
    """
    std_s = np.sqrt(np.diag(cov_s))

    plt.figure(figsize = (10, 6))
    plt.plot(gp.X_train.ravel(), gp.y_train.ravel(), 'ro', label="Training Points")
    plt.plot(X_s, Y_s, 'y-', label="Real Curve")

    plt.plot(X_s, mu_s, 'b-', label="Predicted Curve")
    plt.fill_between(X_s.ravel(), mu_s.ravel()-3*std_s, mu_s.ravel()+3*std_s, alpha=0.2, label="Confidence Level (3σ)")

    plt.title("Gaussian Process Regression")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()


def animate_gp(X_train, y_train, X_s, Y_s, kernel, n, save=False, path="gp_animation.gif", show=True):
    """
    Create an animation showing the iterative update of the GP model as new training points are added.

    Args:
        X_train: (n, 1) - training inputs
        y_train: (n, 1) - training outputs
        X_s: (m, 1) - test inputs
        Y_s: (m, 1) - true function values at test inputs
        kernel: kernel function for the GP
        n: int - number of training points to add iteratively
        save: bool - whether to save the animation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line_mu, = ax.plot([], [], 'b-', label="Predicted Mean")
    line_y, = ax.plot([], [], 'ro', label="Training Points")
    line_true, = ax.plot(X_s, Y_s, 'y-', label="True Function")
    shade = ax.fill_between([], [], [], color='#1f77b4', alpha=0.2, label="Confidence Level (3σ)")

    ax.set_xlim(X_s.min(), X_s.max())
    ax.set_ylim(Y_s.min() - 1, Y_s.max() + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Gaussian Process Regression Iterative Update")
    ax.legend()

    gp_local = None


    def init():
        """
        Initialize the animation.
        """
        line_mu.set_data([], [])
        line_y.set_data([], [])
        return line_mu, line_y

    def update(frame):
        """
        Update function for each frame of the animation.
        """
        print(f"Iteration {frame}")

        nonlocal shade, gp_local
        if not gp_local:
            gp_local = GaussianProcess(X_train[frame].reshape(-1, 1), y_train[frame].reshape(-1, 1), kernel)
        else:
            gp_local.update_L(X_train[frame].reshape(-1, 1), y_train[frame].reshape(-1, 1))

        if frame >= 5:
            gp_local.optimize_hyperparameters()
        mu_s, cov_s = gp_local.stable_gp_predict(X_s)

        # Update training points
        line_y.set_data(X_train, y_train)

        # Update predicted mean
        line_mu.set_data(X_s, mu_s)

        # Update confidence interval shading
        shade.remove()
        std_s = np.sqrt(np.diag(cov_s))
        shade = ax.fill_between(X_s.ravel(), mu_s.ravel()-3*std_s, mu_s.ravel()+3*std_s, color='#1f77b4', alpha=0.2)

        return line_mu, line_y, shade


    anim = FuncAnimation(fig, update, frames=range(n), init_func=init, blit=False, repeat=False)

    if save:
        anim.save(path, writer="ffmpeg", fps=2, dpi=150)

    if show:
        plt.show()