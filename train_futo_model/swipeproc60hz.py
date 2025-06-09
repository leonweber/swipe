"""
Contains the process_data function, which returns ProcessedFeatures.
You can call to_feature_tensor on the returned ProcessedFeatures to get a numpy array.

Use swipe_test_PAN in the swipetest module to verify you are passing correct inputs.
"""

import math
from dataclasses import dataclass
import numpy as np

_ORDER_OF_PROBS = "QWERTYUIOPASDFGHJKLZXCVBNM"

_KEY_WIDTH = 1.0 / 10.0
_KEY_HEIGHT = 1.0 / 4.0
_KEY_CENTERS = {
    "Q": np.array((_KEY_WIDTH * 0.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "W": np.array((_KEY_WIDTH * 1.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "E": np.array((_KEY_WIDTH * 2.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "R": np.array((_KEY_WIDTH * 3.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "T": np.array((_KEY_WIDTH * 4.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "Y": np.array((_KEY_WIDTH * 5.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "U": np.array((_KEY_WIDTH * 6.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "I": np.array((_KEY_WIDTH * 7.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "O": np.array((_KEY_WIDTH * 8.5, _KEY_HEIGHT * 3.5), dtype=np.float32),
    "P": np.array((_KEY_WIDTH * 9.5, _KEY_HEIGHT * 3.5), dtype=np.float32),

    "A": np.array((_KEY_WIDTH * 1.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "S": np.array((_KEY_WIDTH * 2.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "D": np.array((_KEY_WIDTH * 3.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "F": np.array((_KEY_WIDTH * 4.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "G": np.array((_KEY_WIDTH * 5.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "H": np.array((_KEY_WIDTH * 6.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "J": np.array((_KEY_WIDTH * 7.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "K": np.array((_KEY_WIDTH * 8.0, _KEY_HEIGHT * 2.5), dtype=np.float32),
    "L": np.array((_KEY_WIDTH * 9.0, _KEY_HEIGHT * 2.5), dtype=np.float32),

    "Z": np.array((_KEY_WIDTH * 2.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "X": np.array((_KEY_WIDTH * 3.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "C": np.array((_KEY_WIDTH * 4.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "V": np.array((_KEY_WIDTH * 5.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "B": np.array((_KEY_WIDTH * 6.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "N": np.array((_KEY_WIDTH * 7.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
    "M": np.array((_KEY_WIDTH * 8.0, _KEY_HEIGHT * 1.5), dtype=np.float32),
}

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A=1):
    """
    Calculate the pdf of a 2D Gaussian distribution at a given point (x, y).

    Parameters:
    x, y: Coordinates of the point.
    x0, y0: Coordinates of the mean.
    sigma_x, sigma_y: Standard deviations along x and y directions.
    A: Amplitude of the Gaussian.

    Returns:
    The pdf of the 2D Gaussian distribution at the point (x, y).
    """
    return A * math.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))


sigma_x = _KEY_WIDTH/1.6
sigma_y = _KEY_HEIGHT/1.6

NX = 4096
NY = 4096

x_values = np.linspace(0, 1, NX)
y_values = np.linspace(0, 1, NY)
xx, yy = np.meshgrid(x_values, y_values)  # xx, yy each NXxNY arrays

precomputed_gaussians = {}

for key, center in _KEY_CENTERS.items():
    x0, y0 = center
    gvals = np.exp(-(((xx - x0)**2)/(2*sigma_x**2) + ((yy - y0)**2)/(2*sigma_y**2)))
    precomputed_gaussians[key] = gvals

def fast_gaussian_value(key, x, y):
    """
    Return a precomputed approximation of the 2D Gaussian for the given
    key at point (x, y) in [0,1]x[0,1].
    """
    ix = int(round((NX - 1) * x))
    iy = int(round((NY - 1) * y))
    return precomputed_gaussians[key][iy, ix]



_VIRTUAL_KEYBOARD_SIZE = np.array((320, 180))

@dataclass
class ProcessedFeatures:
    """
    The input data gets resampled to maintain a constant distance per step, while keeping sharp corners.

    The following features are calculated trivially: velocity, acceleration, time since gesture start, delta time from last step.

    Angle of approach is calculated by taking the delta in velocity direction.

    Probabilities for each step corresponding to some key or a skip are calculated.
    """

    count: int # total number of points
    x_coords: np.ndarray # 0.0 -> 1.0
    y_coords: np.ndarray # 0.0 -> 1.0 (some points may be out of range)
    times_since_start: np.ndarray # ms
    delta_times: np.ndarray   # ms
    velocity: np.ndarray      # / s
    acceleration: np.ndarray  # / s / 10 ms
    angles: np.ndarray  # angle deltas in radians
    probabilities: dict # "Q": np.ndarray, "W": np.ndarray, ...

    def to_feature_tensor(self) -> np.ndarray:
        """
        Convert this object to a numpy array consisting of model inputs.

        The returned array contains arrays of numbers in the following order:
        x, y, time_since_start, delta_time, velocity_x, velocity_y, acceleration_x, acceleration_y, angle delta,
        probability_q...probability_z
        """
        inputs = []

        for i in range(self.count):
            inputs.append([
                self.x_coords[i],
                self.y_coords[i],
                self.times_since_start[i],
                self.delta_times[i],
                self.velocity[i][0],
                self.velocity[i][1],
                self.acceleration[i][0],
                self.acceleration[i][1],
                self.angles[i],
            ] + [self.probabilities[c][i] for c in _ORDER_OF_PROBS])

        if len(inputs) == 0:
            return np.zeros((0, 9 + len(_ORDER_OF_PROBS))) # to ensure correct shape

        return np.array(inputs)

import time

def process_data(norm_x_coords, norm_y_coords, timestamps, do_resampling_insert=True, strict=False):
    """
    Data formatting requirements

    All coordinates must be normalized between 0 and 1. The following coordinate system is used:

    ^ +y                 
    |                    The bottom left corner is (0, 0)
    |                    The top left corner is (0, 1)
    |                    The bottom right corner is (1, 0)
    |                    The top right corner is (1, 1)
    ----------->   +x


    The keyboard is assumed to be four rows in the following layout:

    Q W E R T Y U I O P
     A S D F G H J K L
       Z X C V B N M
        space

    It thus follows that (0.5, 0.125) sits on the spacebar, and (0.95, 0.875) sits on the letter P


    Timestamps must be specified in milliseconds. Either relative timestamps where the times start from 0, or UNIX timestamps.
    The verifier checks the UNIX timestamps are between the years 2010 and 2030.
    """

    # Lengths must be equal in all cases
    assert len(norm_x_coords) == len(norm_y_coords) == len(timestamps)

    if strict:
        # Coordinates must be normalized
        assert (norm_x_coords >= 0.0).all() and (norm_y_coords >= 0.0).all()
        assert (norm_x_coords <= 1.0).all() and (norm_y_coords <= 1.0).all()

        # Timestamp must be either a UNIX timestamp in milliseconds, or a relative timestamp starting from zero
        is_unix_ms_timestamp = (timestamps >= 1262350861 * 1000).all() and (timestamps <= 1893502861 * 1000).all()
        is_relative_timestamp = (timestamps[0] == 0) and (timestamps[-1] < 30 * 1000)
        assert is_unix_ms_timestamp or is_relative_timestamp

        # Types must be correct
        assert (norm_x_coords.dtype == np.float32) or (norm_x_coords.dtype == np.float64)
        assert (norm_y_coords.dtype == np.float32) or (norm_y_coords.dtype == np.float64)
        assert (timestamps.dtype == np.int64)
    

    # 1. Resample the coords to maintain consistent density
    norm_x_coords, norm_y_coords, timestamps = _resample_coordinates(norm_x_coords, norm_y_coords, timestamps, do_resampling_insert=do_resampling_insert)

    # 2. Calculate times
    times_since_start = np.array(list(map(lambda x: x - timestamps[0], timestamps)), dtype=np.int32)

    delta_times = [0]
    for i in range(1, len(times_since_start)):
        delta_times.append(times_since_start[i] - times_since_start[i-1])

    delta_times = np.fmax(np.array(delta_times, dtype=np.int32), 1)

    # 3. Calculate velocities and accelerations
    velocities = [(0.0, 0.0)]
    for i in range(1, len(norm_x_coords)):
        delta_x = norm_x_coords[i] - norm_x_coords[i - 1]
        delta_y = norm_y_coords[i] - norm_y_coords[i - 1]
        delta_t = delta_times[i] / 1000.0

        velocities.append((
            delta_x / delta_t,
            delta_y / delta_t
        ))
    
    accelerations = [(0.0, 0.0)]
    for i in range(1, len(velocities)):
        delta_delta_x = velocities[i][0] - velocities[i - 1][0]
        delta_delta_y = velocities[i][1] - velocities[i - 1][1]
        delta_t = delta_times[i] / 10.0

        accelerations.append((
            delta_delta_x / delta_t,
            delta_delta_y / delta_t
        ))

    # 4. Calculate angles
    angles = [0.0]
    for i in range(2, len(velocities)):
        v0 = velocities[i-1]
        v1 = velocities[i]

        angle0 = math.atan2(v0[1], v0[0])
        angle1 = math.atan2(v1[1], v1[0])

        angles.append(math.atan2(math.sin(angle1 - angle0), math.cos(angle1 - angle0)))

    angles.append(angles[-1])

    # 5. Calculate probabilities
    i_x = np.round((NX - 1) * norm_x_coords.clip(0, 1)).astype(int)  # shape (N,)
    i_y = np.round((NY - 1) * norm_y_coords.clip(0, 1)).astype(int)  # shape (N,)

    probabilities = {}
    for c in _KEY_CENTERS.keys():
        probabilities[c] = precomputed_gaussians[c][i_y, i_x]  # shape (N,)
        #all_probs_for_key_c = 


    #for i in range(len(norm_x_coords)):
    #    x, y = norm_x_coords[i], norm_y_coords[i]

    #    for c in _KEY_CENTERS.keys():
            
            #kx, ky = _KEY_CENTERS[c]
    #        probabilities[c].append(
    #            #gaussian_2d(x, y, kx, ky, _KEY_WIDTH/1.6, _KEY_HEIGHT/1.6)
    #            fast_gaussian_value(c, x, y)
    #        )


    return ProcessedFeatures(
        count = len(norm_x_coords),
        x_coords = norm_x_coords,
        y_coords = norm_y_coords,
        times_since_start = times_since_start,
        delta_times = delta_times,
        velocity = np.array(velocities, dtype=np.float32),
        acceleration = np.array(accelerations, dtype=np.float32),
        angles = angles,
        probabilities = probabilities
    )

def _insert_inbetween_points(x, y, t):
    """
    Makes the input more dense, so that the resampling fuction can ensure
    a consistently dense output no matter how sparse the input is
    """
    # TODO: Linear time interpolation doesn't take into account acceleration,
    # all added points have 0 acceleration

    kw = _VIRTUAL_KEYBOARD_SIZE[0]
    kh = _VIRTUAL_KEYBOARD_SIZE[1]
    max_dist = kw * _KEY_WIDTH * 0.6

    new_data = [(x[0], y[0], t[0])]

    for i in range(1, len(x)):
        delta_x = (x[i] * kw) - (x[i-1] * kw)
        delta_y = (y[i] * kh) - (y[i-1] * kh)
        delta_t = t[i] - t[i-1]

        delta_length = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        current_t = delta_length
        while current_t > 0:
            current_t = max(0, current_t - max_dist)
            mult = (delta_length - current_t) / delta_length
            
            inbetween_x = (x[i-1] * kw) + delta_x * mult
            inbetween_y = (y[i-1] * kh) + delta_y * mult
            inbetween_t = t[i-1]        + delta_t * mult

            new_data.append((
                inbetween_x / kw,
                inbetween_y / kh,
                inbetween_t
            ))
    
    return (
        list(map(lambda x: x[0], new_data)),
        list(map(lambda x: x[1], new_data)),
        list(map(lambda x: x[2], new_data))
    )



RESAMPLE_HZ = 60
def _resample_coordinates(x, y, t, do_resampling_insert):
    """
    Resample given coordinates, and normalize time to start from 0
    """

    tStart = t[0]
    tEnd = t[-1]
    duration = tStart - tEnd

    dt = 1000 / RESAMPLE_HZ
    uniformTimestamps = []

    time = tStart
    while time <= tEnd:
        uniformTimestamps.append(time)
        time += dt


    resampledX = []
    resampledY = []
    resampledT = []

    for time in uniformTimestamps:
        for i, pointT in enumerate(t):
            if pointT > time: break

        if pointT <= time:
            # If `t` exceeds last point, just use the last value
            resampledX.append(x[-1])
            resampledY.append(y[-1])
            resampledT.append(t[-1] - t[0])
        else:
            p1x = x[i - 1]
            p1y = y[i - 1]
            p1t = t[i - 1]
            p2x = x[i]
            p2y = y[i]
            p2t = t[i]

            ratio = (time - p1t) / (p2t - p1t)
            interpolatedX = p1x + ratio * (p2x - p1x)
            interpolatedY = p1y + ratio * (p2y - p1y)

            resampledX.append(interpolatedX)
            resampledY.append(interpolatedY)
            resampledT.append(time - t[0])
    return (
        np.array(resampledX, dtype=np.float32),
        np.array(resampledY, dtype=np.float32),
        np.array(resampledT, dtype=np.int64),
    )
    """
    if do_resampling_insert:
        x, y, t = _insert_inbetween_points(x, y, t) # Make sparse paths more dense

    kw = _VIRTUAL_KEYBOARD_SIZE[0]
    kh = _VIRTUAL_KEYBOARD_SIZE[1]
    thresh = kw * _KEY_WIDTH * 1.65618281828
    angle_thresh = 0.9

    new_data = [(x[0], y[0], 0)]

    curr_distance = 0
    prev_angle = None
    curr_angle_delta = 0
    for i in range(1, len(x)):
        delta_x = (x[i] * kw) - (x[i-1] * kw)
        delta_y = (y[i] * kh) - (y[i-1] * kh)
        delta_t = t[i] - (new_data[-1][2] + t[0])

        if abs(delta_x) > 0.0 or abs(delta_y) > 0.0:
            angle = math.atan2(delta_y, delta_x)
            if prev_angle:
                curr_angle_delta += abs(angle - prev_angle)
            prev_angle = angle

        curr_distance += math.sqrt(delta_x * delta_x + delta_y * delta_y)

        exceeds_distance_thresh = curr_distance > thresh
        exceeds_angle_thresh = curr_angle_delta > angle_thresh
        is_final_point = i+1 == len(x)

        if exceeds_angle_thresh and delta_t > 0.0:
            if curr_distance < thresh*0.08:
                new_data.pop(-1)
            curr_distance = math.sqrt(delta_x * delta_x + delta_y * delta_y) # Keep current distance
            curr_angle_delta = 0.0
            new_data.append((x[i-1], y[i-1], t[i-1] - t[0]))
            
        elif (exceeds_distance_thresh or is_final_point) and delta_t > 0.0:
            # Remove the penultimate point if it's too close
            if is_final_point and curr_distance < thresh*0.4:
                new_data.pop(-1)

            curr_distance = 0
            curr_angle_delta = 0.0
            new_data.append((x[i], y[i], t[i] - t[0]))
    
    return (
        np.array(list(map(lambda x: x[0], new_data)), dtype=np.float32),
        np.array(list(map(lambda x: x[1], new_data)), dtype=np.float32),
        np.array(list(map(lambda x: x[2], new_data)), dtype=np.int64),
    )"""