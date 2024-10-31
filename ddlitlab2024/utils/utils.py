import numpy as np
from transforms3d.quaternions import quat2axangle


def quats_to_5d(quats: np.ndarray) -> np.ndarray:
    """
    Convert an array of quaternions (xyzw) to 5D representations (sin, cos, x, y, z).
    """
    # Convert to (wxyz) representation (required by quat2axangle)
    quats = xyzw2wxyz(quats)

    # Apply the conversion
    vectors, angles = map(np.array, zip(*map(quat2axangle, quats)))

    # Make continuous angle representation
    angle_sin = np.sin(angles)[: , None]
    angle_cos = np.cos(angles)[: , None]

    # Build the 5D representation array
    return np.concatenate((vectors, angle_sin, angle_cos), axis=-1)

def xyzw2wxyz(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (xyzw) to (wxyz) representation.

    :param quat: The quaternion.
    :return: The (wxyz) representation.
    """
    return np.roll(quat, 1, axis=-1)

def wxyz2xyzw(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (wxyz) to (xyzw) representation.

    :param quat: The quaternion.
    :return: The (xyzw) representation.
    """
    return np.roll(quat, -1, axis=-1)
