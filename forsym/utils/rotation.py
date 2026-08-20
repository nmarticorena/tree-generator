import numpy as np


def matrix_to_rpy(matrix):
    """Convert a rotation matrix to URDF roll-pitch-yaw coordinates.

    Parameters
    ----------
    matrix : array-like, shape (3, 3)
        Three-dimensional rotation matrix.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Roll, pitch, and yaw in radians.
    """
    matrix = np.asanyarray(matrix)
    if np.abs(matrix[2, 0]) >= 1.0 - 1e-12:
        yaw = 0.0
        if matrix[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(matrix[0, 1], matrix[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-matrix[0, 1], -matrix[0, 2])
    else:
        pitch = -np.arcsin(matrix[2, 0])
        roll = np.arctan2(matrix[2, 1] / np.cos(pitch), matrix[2, 2] / np.cos(pitch))
        yaw = np.arctan2(matrix[1, 0] / np.cos(pitch), matrix[0, 0] / np.cos(pitch))
    return np.array([roll, pitch, yaw])


def get_rotation_matrix(vec1, vec2):
    """Calculate the rotation that aligns one direction with another.

    Parameters
    ----------
    vec1 : numpy.ndarray
        Source direction vector.
    vec2 : numpy.ndarray
        Target direction vector.

    Returns
    -------
    numpy.ndarray
        Three-by-three rotation matrix.
    """
    vec2 = vec2 / np.linalg.norm(vec2)
    vec1 = vec1 / np.linalg.norm(vec1)

    axis = np.cross(vec1, vec2)
    cosine = np.dot(vec1, vec2)

    x, y, z = axis
    scale = 1 / (1 + cosine)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    return np.eye(3, dtype=np.float64) + skew + skew.dot(skew) * scale


def calculate_rpy(l1_start, l1_end, l2_start, l2_end):
    """Calculate the rotation between two directed line segments.

    Parameters
    ----------
    l1_start : forsym.fractal.turtle.Point
        Start of the first segment.
    l1_end : forsym.fractal.turtle.Point
        End of the first segment.
    l2_start : forsym.fractal.turtle.Point
        Start of the second segment.
    l2_end : forsym.fractal.turtle.Point
        End of the second segment.

    Returns
    -------
    tuple of float
        Roll, pitch, and yaw in radians. Degenerate segments return zeros.
    """

    first = np.array([l1_end.x - l1_start.x, l1_end.y - l1_start.y, l1_end.z - l1_start.z])
    second = np.array([l2_end.x - l2_start.x, l2_end.y - l2_start.y, l2_end.z - l2_start.z])
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm < 1e-6 or second_norm < 1e-6:
        return 0, 0, 0

    matrix = get_rotation_matrix(second / second_norm, first / first_norm)
    return tuple(matrix_to_rpy(matrix))
