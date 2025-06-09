import numpy as np


def sample_trajectory(
    trajectory: np.ndarray,
    num_samples: int,
    noise_std: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Sample points equidistantly from the interpolated trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to sample from, shape (num_points, n_dims).
    num_samples : int
        The number of samples to take.
    noise_std : float | np.ndarray, optional
        The standard deviation of the noise to add to the trajectory.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, n_dims) containing the equidistantly sampled points.
    """
    # Convert to float to prevent type errors when adding noise
    trajectory = trajectory.astype(float).copy()

    if (
        isinstance(noise_std, np.ndarray)
        and noise_std.shape[0] != trajectory.shape[1]
    ):
        raise ValueError(
            "`noise_std` must have the same shape as `trajectory`"
        )

    if num_samples <= 0:
        return np.zeros((num_samples, trajectory.shape[1]))

    # Convert scalar noise_std to array if needed
    if isinstance(noise_std, float):
        noise_std = np.array([noise_std] * trajectory.shape[1])

    # Add noise if any dimension has non-zero noise
    if not (noise_std == 0).all():
        for idx_dim in range(trajectory.shape[1]):
            if noise_std[idx_dim] > 0:  # Only add noise if std > 0
                trajectory[:, idx_dim] += np.random.normal(
                    0, noise_std[idx_dim], trajectory.shape[0]
                )

        if len(trajectory) <= 1:
            # If trajectory has 0 or 1 point, return that point repeated num_samples times
            return (
                np.tile(trajectory[0], (num_samples, 1))
                if len(trajectory) == 1
                else np.zeros(
                    (
                        num_samples,
                        trajectory.shape[1] if trajectory.ndim > 1 else 0,
                    )
                )
            )

    # Calculate segment lengths
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Handle cases where segments have zero length (consecutive identical points)
    # This prevents division by zero later
    # though the interpolation logic should handle it if cumulative_distances are correct.
    # A more robust way is to filter out zero-length segments or handle them explicitly.
    # For simplicity, we'll let the cumulative_distances handle it, and the interpolation factor
    # will be 0 if the segment length is 0 and distance_sample matches cumulative_distances[idx_start].

    # Calculate cumulative distances
    # Prepend 0.0 for the start of the first segment
    cumulative_distances = np.concatenate(([0.0], np.cumsum(distances)))

    total_length = cumulative_distances[-1]

    # If the total length is zero (all points are identical), return the first point repeated
    if total_length == 0:
        return np.tile(trajectory[0], (num_samples, 1))

    sampled_trajectory = np.zeros((num_samples, trajectory.shape[1]))

    # The first sampled point is always the first point of the trajectory
    sampled_trajectory[0] = trajectory[0]

    # Calculate the target distances for sampling
    # We want num_samples points, so we divide the total length into num_samples - 1 intervals
    # The target distances will range from 0 to total_length
    target_distances = np.linspace(0, total_length, num_samples)

    # Iterate through each target distance to find the corresponding interpolated point
    for i in range(1, num_samples):
        distance_sample = target_distances[i]

        # Find the index of the segment where the distance_sample falls
        # np.searchsorted returns the index where distance_sample would be inserted
        # to maintain order. This means cumulative_distances[idx_segment] <= distance_sample < cumulative_distances[idx_segment + 1]
        idx_segment = (
            np.searchsorted(
                cumulative_distances, distance_sample, side="right"
            )
            - 1
        )

        # Ensure idx_segment is within valid bounds for accessing trajectory points
        # If distance_sample is exactly total_length, idx_segment might be len(cumulative_distances) - 1
        # In that case, we take the last point of the trajectory.
        if idx_segment >= len(trajectory) - 1:
            sampled_trajectory[i] = trajectory[-1]
            continue

        # Get the start and end points of the current segment
        start_point = trajectory[idx_segment]
        end_point = trajectory[idx_segment + 1]

        # Get the cumulative distance at the start of the segment
        dist_at_start_segment = cumulative_distances[idx_segment]

        # Calculate the length of the current segment
        segment_length = distances[
            idx_segment
        ]  # This is the actual length, not normalized

        # Calculate the interpolation factor (t)
        # t = 0 means at start_point, t = 1 means at end_point
        if segment_length == 0:
            # If segment has zero length, the point is the start_point (or end_point, they are identical)
            interpolation_factor = (
                0.0  # Or 1.0, doesn't matter as start_point == end_point
            )
        else:
            interpolation_factor = (
                distance_sample - dist_at_start_segment
            ) / segment_length

        # Interpolate the point
        interpolated_point = start_point + interpolation_factor * (
            end_point - start_point
        )
        sampled_trajectory[i] = interpolated_point

    return sampled_trajectory
