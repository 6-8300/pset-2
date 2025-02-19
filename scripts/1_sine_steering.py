import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.sine import angle_to_coeffs, coeffs_to_sine


def sine_steering():
    # Create the domain for x using torch for consistency with student functions.
    x = torch.linspace(0, 2 * np.pi, 200)
    x_np = x.numpy()  # for plotting with matplotlib

    # Set up the figure and axes.
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize three lines: one for each component and the sum.
    (line_a,) = ax.plot(x_np, np.zeros_like(x_np), label=r"$a\cos(x)$", color="C0")
    (line_b,) = ax.plot(x_np, np.zeros_like(x_np), label=r"$b\sin(x)$", color="C1")
    (line_sum,) = ax.plot(
        x_np, np.zeros_like(x_np), label=r"$a\cos(x)+b\sin(x)$", color="C2", linewidth=2
    )

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper right")

    # Define a set of angles that sweep over the unit circle.
    angles = np.linspace(0, 2 * np.pi, 60)

    def update(frame):
        # For each frame, compute the coefficients from the current angle.
        angle = angles[frame]
        a, b = angle_to_coeffs(torch.tensor(angle))

        # Compute the individual components and the summed sine wave.
        # Using the provided student function for the sum.
        y_a = a * torch.cos(x)
        y_b = b * torch.sin(x)
        y_sum = coeffs_to_sine(a, b, x)

        # Update the plot lines (convert torch tensors to numpy arrays).
        line_a.set_ydata(y_a.numpy())
        line_b.set_ydata(y_b.numpy())
        line_sum.set_ydata(y_sum.numpy())

        ax.set_title(f"Sine Steering (a = {a:.2f}, b = {b:.2f})")
        return line_a, line_b, line_sum

    ani = animation.FuncAnimation(
        fig, update, frames=len(angles), interval=100, blit=True
    )

    # Save the animation as a gif using the Pillow writer.
    ani.save("img/sine_steering.gif", writer="pillow", fps=10)
    plt.close(fig)


if __name__ == "__main__":
    sine_steering()
