import os
import numpy as np
import matplotlib.pyplot as plt


class CircleDrawer:
    def __init__(self, fig, ax, save_path="strokes.npy"):
        """
        Initialize the CircleDrawer with figure, axis, and save path.

        Args:
            fig: Matplotlib figure object.
            ax: Matplotlib axis object.
            save_path: Path to save the strokes data.
        """
        self.strokes = []
        self.current_x = []
        self.current_y = []

        self.fig = fig
        self.ax = ax

        self.save_path = save_path

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        """
        Start a new stroke on mouse press.
        """
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.current_x = [event.xdata]
            self.current_y = [event.ydata]

    def on_motion(self, event):
        """
        Update the stroke on mouse movement.
        """
        if len(self.current_x) == 0:
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.current_x.append(event.xdata)
            self.current_y.append(event.ydata)
            self.ax.plot(self.current_x, self.current_y, color='blue')
            self.fig.canvas.draw()

    def on_release(self, event):
        """
        Complete the stroke on mouse release.
        """
        if event.button != 1:
            return
        
        if len(self.current_x) > 1:
            curr_stroke = np.column_stack((self.current_x, self.current_y))
            self.strokes.append(curr_stroke)
            np.save(self.save_path, np.array(self.strokes, dtype=object))
            print(f"Saved {len(self.strokes)} strokes to {self.save_path}. Last stroke length: {len(curr_stroke)} points.")
        
        self.current_x = []
        self.current_y = []


# Path to save the strokes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "..", "data", "circle", "strokes.npy")

# Set up the plot
fig, ax = plt.subplots()
ax.set_title('Draw a Circle with Mouse')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

"""
ax.plot(0, 2, 'ro')
ax.plot(0, -2, 'ro')
ax.plot(2, 0, 'ro')
ax.plot(-2, 0, 'ro')
"""

# Initialize CircleDrawer
circle_drawer = CircleDrawer(fig, ax, save_path=path)

plt.show()