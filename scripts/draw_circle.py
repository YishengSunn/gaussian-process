import numpy as np
import matplotlib.pyplot as plt


class CircleDrawer:
    def __init__(self, fig, ax):
        """
        Initialize the CircleDrawer with figure and axis
        strokes: list of completed strokes
        current_x, current_y: current stroke coordinates
        """
        self.strokes = []
        self.current_x = []
        self.current_y = []

        self.fig = fig
        self.ax = ax

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        """Start a new stroke on mouse press."""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.current_x = [event.xdata]
            self.current_y = [event.ydata]

    def on_motion(self, event):
        """Add points to the current stroke on mouse movement."""
        if len(self.current_x) == 0:
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.current_x.append(event.xdata)
            self.current_y.append(event.ydata)
            self.ax.plot(self.current_x, self.current_y, color='blue')
            self.fig.canvas.draw()

    def on_release(self, event):
        """Complete the stroke on mouse release."""
        if event.button != 1:
            return
        
        if len(self.current_x) > 1:
            self.strokes.append((self.current_x, self.current_y))
        
        self.current_x = []
        self.current_y = []

        print(f"Completed strokes: {len(self.strokes)}, Last stroke length: {len(self.strokes[-1][0]) if self.strokes else 0}")

fig, ax = plt.subplots()
ax.set_title('Draw a Circle with Mouse')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

circle_drawer = CircleDrawer(fig, ax)

plt.show()