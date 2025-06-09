import numpy as np
import pytest
from train_futo_model.util import sample_trajectory


def test_basic_2d_trajectory():
    """Test basic functionality with a simple 2D trajectory."""
    # Create a simple diagonal line
    trajectory = np.array([[0, 0], [1, 1], [2, 2]])

    sampled = sample_trajectory(trajectory, num_samples=3)

    # Should have the correct shape
    assert sampled.shape == (3, 2)
    # First and last points should match
    np.testing.assert_array_almost_equal(sampled[0], [0, 0])
    np.testing.assert_array_almost_equal(sampled[-1], [2, 2])
    # Middle point should be equidistant
    np.testing.assert_array_almost_equal(sampled[1], [1, 1], decimal=5)


def test_single_point():
    """Test with a trajectory containing only one point."""
    trajectory = np.array([[1, 2, 3]])
    sampled = sample_trajectory(trajectory, num_samples=5)

    assert sampled.shape == (5, 3)
    # All points should be the same as the input
    for point in sampled:
        np.testing.assert_array_almost_equal(point, [1, 2, 3])


def test_same_number_of_samples():
    """Test when num_samples equals the number of points."""
    trajectory = np.array([[0, 0], [1, 2], [3, 6]])
    sampled = sample_trajectory(trajectory, num_samples=3)
    expected_trajectory = np.array([[0, 0], [1.5, 3], [3, 6]])

    assert sampled.shape == (3, 2)
    # Points should be very close to original (might have small floating
    # point differences)
    np.testing.assert_array_almost_equal(
        sampled, expected_trajectory, decimal=5
    )


def test_more_samples_than_points():
    """Test with more samples than original points."""
    trajectory = np.array([[0, 0], [1, 1]])
    num_samples = 5
    sampled = sample_trajectory(trajectory, num_samples=num_samples)

    assert sampled.shape == (num_samples, 2)
    # First and last points should match exactly
    np.testing.assert_array_almost_equal(sampled[0], [0, 0])
    np.testing.assert_array_almost_equal(sampled[-1], [1, 1])
    # Points should be equidistant
    for i in range(1, num_samples):
        dist1 = np.linalg.norm(sampled[i] - sampled[i - 1])
        dist2 = np.linalg.norm(sampled[1] - sampled[0])
        np.testing.assert_almost_equal(dist1, dist2, decimal=5)


def test_3d_trajectory():
    """Test with a 3D trajectory."""
    trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    sampled = sample_trajectory(trajectory, num_samples=4)

    assert sampled.shape == (4, 3)
    # Check first and last points
    np.testing.assert_array_almost_equal(sampled[0], [0, 0, 0])
    np.testing.assert_array_almost_equal(sampled[-1], [2, 2, 2])
    # Check that all dimensions are equal (diagonal line in 3D)
    for point in sampled:
        assert np.allclose(point[0], point[1])
        assert np.allclose(point[1], point[2])


def test_gaussian_noise_single_std():
    """Test adding Gaussian noise with a single standard deviation."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Simple 2D line with more points to better test noise
    trajectory = np.array([[0, 0], [5, 5], [10, 10]])

    # Add noise with a small std since we're testing the noise addition
    noise_std = 0.1

    # Run multiple trials to get more stable statistics
    all_distances = []
    for _ in range(100):
        sampled = sample_trajectory(
            trajectory, num_samples=10, noise_std=noise_std
        )
        # Calculate distances from the line y = x
        distances = np.abs(sampled[:, 1] - sampled[:, 0]) / np.sqrt(2)
        all_distances.extend(distances)

    # The standard deviation of the distances should be close to the noise_std

    # We use a tolerance because with a small number of samples, the empirical
    # std might differ. For a 2D Gaussian with equal std in x and y, the
    # perpendicular distance has $\sigma_{\text{perpendicular}} = 0.6028 \cdot \sigma $
    expected_std = noise_std * 0.6028
    measured_std = np.std(all_distances)

    assert measured_std > 0
    assert measured_std == pytest.approx(expected_std, rel=0.2)


def test_gaussian_noise_per_dimension():
    """Test noise with different standard deviations per dimension."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Simple 2D line with more points to better test noise
    trajectory = np.array([[0, 0], [5, 5], [10, 10]])

    # Different noise levels for x and y dimensions
    noise_std = np.array([0.1, 0.4])  # y has 4x more noise than x

    # Run multiple trials to get more stable statistics
    all_x_noise = []
    all_y_noise = []

    for _ in range(100):
        sampled = sample_trajectory(
            trajectory, num_samples=10, noise_std=noise_std
        )

        # Calculate noise in each dimension
        expected_line = np.linspace(0, 10, 10)
        noise_x = sampled[:, 0] - expected_line
        noise_y = sampled[:, 1] - expected_line

        all_x_noise.extend(noise_x)
        all_y_noise.extend(noise_y)

    # Calculate standard deviations
    std_x = np.std(all_x_noise)
    std_y = np.std(all_y_noise)

    # Check that y has more variation than x
    assert std_y > std_x, f"Expected y-std ({std_y}) > x-std ({std_x})"

    # Check that the ratio of stds is in the expected direction
    # The actual ratio will be less than the input ratio because noise is only
    # added to the original points, not the interpolated ones
    assert std_y > std_x, f"Expected y-std ({std_y}) > x-std ({std_x})"

    # The ratio should be greater than 1.0 since we added more noise to y
    # But it won't be as large as the input ratio due to interpolation
    assert std_y / std_x > 1.5, (
        f"Expected y-std > 1.5 * x-std, got ratio {std_y / std_x:.2f}"
    )
