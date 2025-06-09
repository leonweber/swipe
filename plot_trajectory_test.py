import numpy as np
import matplotlib.pyplot as plt
from train_futo_model.util.geometry import sample_trajectory


def plot_trajectory_test():
    # Original trajectory points
    trajectory = np.array([[0, 0], [1, 2], [3, 6]])

    # Expected sampled points
    expected_trajectory = np.array([[0, 0], [1.5, 3], [3, 6]])

    # Get the actual sampled points
    sampled = sample_trajectory(trajectory, num_samples=3)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot original trajectory points
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        "bo-",
        label="Original Trajectory",
        alpha=0.5,
        linewidth=2,
    )

    # Plot expected sampled points
    plt.plot(
        expected_trajectory[:, 0],
        expected_trajectory[:, 1],
        "ro--",
        label="Expected Sampled Points",
        linewidth=2,
        markersize=8,
    )

    # Plot actual sampled points
    plt.plot(
        sampled[:, 0],
        sampled[:, 1],
        "gx--",
        label="Actual Sampled Points",
        linewidth=2,
        markersize=10,
        mew=2,
    )

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Sampling Test")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Annotate the points
    for i, (x, y) in enumerate(trajectory):
        plt.annotate(
            f"P{i + 1}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_trajectory_test()
