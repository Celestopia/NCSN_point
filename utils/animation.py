import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def make_point_animation(fig, ax, points, size=5, color='blue', frame_text_func=lambda frame: f"Frame {frame}"):
    """
    Make an animation of a set of 2-d points over time.

    Args:
        fig (matplotlib.figure.Figure): The figure object to plot the animation on.
        ax (matplotlib.axes.Axes): The axes object to plot the animation on.
        points (np.ndarray): An array of shape (T, n_samples, 2) containing the coordinates of the points in each frame.
        size (int): The size of the points to plot.
        color (str): The color of the points to plot.
        frame_text_func (function): A function that takes a frame number (int) and returns the text to display for that frame.
    
    Returns:
        ani (matplotlib.animation.FuncAnimation): The animation object.

    Example usage:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        points = np.random.randn(200, 100, 2) # Generate 200 frames of 100 points in 2-d space
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        frame_text_func = lambda frame: f"Frame {frame}/{len(points)}"
        ani = make_point_animation(fig, ax, points, frame_text_func=frame_text_func)
        ax.scatter([0],[0], c='r', s=20) # You can still add other elements to the plot
        save_path = 'animation111.gif'
        ani.save(save_path, writer='pillow', fps=30) # Save animation as gif
        plt.show()
        ```
    """
    scatter = ax.scatter([], [], s=size, c=color)

    frame_text = ax.text(0.5, 0.95, "",
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

    def init():
        scatter.set_offsets([points[0]])
        return scatter,

    def update(frame):
        scatter.set_offsets(points[frame]) # Update scatter object with current frame data
        frame_text.set_text(frame_text_func(frame))
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(points), init_func=init, blit=True)

    return ani




if __name__ == '__main__':
    points = np.random.randn(200, 100, 2) # Generate 200 frames of 100 points in 2-d space
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    frame_text_func = lambda frame: f"Frame {frame}/{len(points)}"
    ani = make_point_animation(fig, ax, points, frame_text_func=frame_text_func)
    ax.scatter([0],[0], c='r', s=20) # You can still add other elements to the plot
    save_path = 'animation111.gif'
    ani.save(save_path, writer='pillow', fps=30) # Save animation as gif
    plt.show()

