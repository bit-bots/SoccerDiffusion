import re

import numpy as np
from transforms3d.quaternions import quat2axangle

CAMELCASE_TO_SNAKECASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def quats_to_5d(quats: np.ndarray) -> np.ndarray:
    """
    Convert an array of quaternions (xyzw) to 5D representations (sin, cos, x, y, z).
    """
    # Convert to (wxyz) representation (required by quat2axangle)
    quats = xyzw2wxyz(quats)

    # Apply the conversion
    vectors, angles = map(np.array, zip(*map(quat2axangle, quats)))

    # Make continuous angle representation
    angle_sin = np.sin(angles)[:, None]
    angle_cos = np.cos(angles)[:, None]

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


def shift_radian_to_positive_range(radian: float) -> float:
    """
    Shift a principal range radian [-pi, pi] to the positive principal range [0, 2pi].

    :param radian: The pricipal range radian radian [-pi, pi].
    :return: The positive principal range radian [0, 2pi].
    """
    return (radian + 3 * np.pi) % (2 * np.pi)


def timestamp_in_ns(seconds: int, nanoseconds: int) -> int:
    """
    Convert a combined unix timestamp from seconds and nanoseconds to timestamp in nanoseconds.
    """
    return int(seconds * 1e9) + nanoseconds


def timestamp_in_s(seconds: int, nanoseconds: int) -> float:
    """
    Convert a timestamp in nanoseconds to a combined unix timestamp in seconds as float.
    """
    return seconds + nanoseconds / 1e9


def camelcase_to_snakecase(name: str) -> str:
    """
    Convert a camelCase string to snake_case.
    """
    return CAMELCASE_TO_SNAKECASE_REGEX.sub("_", name).lower()
