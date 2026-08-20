"""Three-dimensional heading-left-up operations for the L-system turtle."""

import math

import numpy as np


def rotate_around_u(angle, hlu):
    """Rotate the turtle frame around its up axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    hlu : tuple of array-like
        Heading, left, and up vectors.

    Returns
    -------
    tuple of list
        Rotated heading, left, and up vectors.
    """
    hlu = np.array(hlu).T

    rotation_matrix = np.array(
        [[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def rotate_around_h(angle, hlu):
    """Rotate the turtle frame around its heading axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    hlu : tuple of array-like
        Heading, left, and up vectors.

    Returns
    -------
    tuple of list
        Rotated heading, left, and up vectors.
    """
    hlu = np.array(hlu).T

    rotation_matrix = np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def rotate_around_l(angle, hlu):
    """Rotate the turtle frame around its left axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    hlu : tuple of array-like
        Heading, left, and up vectors.

    Returns
    -------
    tuple of list
        Rotated heading, left, and up vectors.
    """
    hlu = np.array(hlu).T

    rotation_matrix = np.array(
        [[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]]
    )
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def keep_l_horizontal(hlu):
    """Realign the turtle frame so its left axis is horizontal.

    Parameters
    ----------
    hlu : tuple of array-like
        Heading, left, and up vectors.

    Returns
    -------
    tuple of list
        Realigned heading, left, and up vectors.
    """
    H = hlu[0]
    [xh, yh, zh] = H
    L = [-yh, -xh, 0]
    U = [xh * zh, -zh * yh, -(xh**2) - yh**2]
    return H, L, U


def normalize(vector):
    """Scale a vector to unit length.

    Parameters
    ----------
    vector : array-like
        Vector to normalize.

    Returns
    -------
    array-like
        Unit vector, or the original zero vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def torque(hlu, tropism):
    """Calculate the tropism torque on the turtle heading.

    Parameters
    ----------
    hlu : tuple of array-like
        Heading, left, and up vectors.
    tropism : array-like
        Three-dimensional tropism vector.

    Returns
    -------
    list of float
        Cross product of the heading and tropism vectors.
    """
    H = hlu[0]
    [xh, yh, zh] = H
    [xt, yt, zt] = tropism
    torque = [yh * zt - zh * yt, zh * xt - xh * zt, xh * yt - yh * xt]
    return torque


def rotate(hlu, axis, angle):
    """Rotate the turtle frame around an arbitrary axis.

    Parameters
    ----------
    hlu : tuple of array-like
        Heading, left, and up vectors.
    axis : array-like
        Three-dimensional rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    tuple of list
        Rotated heading, left, and up vectors.
    """
    axis = normalize(axis)
    [ax, ay, az] = axis
    c = math.cos(angle)
    s = math.sin(angle)
    rotation_matrix = np.array(
        [
            [ax**2 * (1 - c) + c, ax * ay * (1 - c) - az * s, ax * az * (1 - c) + ay * s],
            [ax * ay * (1 - c) + az * s, ay**2 * (1 - c) + c, ay * az * (1 - c) - ax * s],
            [ax * az * (1 - c) - ay * s, ay * az * (1 - c) + ax * s, az**2 * (1 - c) + c],
        ]
    )
    [H, L, U] = hlu
    H = np.array(H)
    L = np.array(L)
    U = np.array(U)
    H = np.dot(rotation_matrix, H)
    L = np.dot(rotation_matrix, L)
    U = np.dot(rotation_matrix, U)
    return H.tolist(), L.tolist(), U.tolist()


def apply_tropism(hlu, tropism):
    """Rotate the turtle frame toward a tropism vector.

    Parameters
    ----------
    hlu : tuple of array-like
        Heading, left, and up vectors.
    tropism : array-like
        Three-dimensional tropism vector.

    Returns
    -------
    tuple of list
        Tropism-adjusted heading, left, and up vectors.
    """
    axis = torque(hlu, tropism)
    angle = np.linalg.norm(axis)
    return rotate(hlu, axis, angle)
