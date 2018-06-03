import numpy as np


def angle_with_plane(a, n):
    a = np.asarray(a)
    n = np.asarray(n)

    a_norm = a / np.linalg.norm(a)
    n_norm = n / np.linalg.norm(n)

    return np.arcsin(np.dot(a, n))


def shortest_angle_between(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    return np.arccos(np.dot(a_norm, b_norm))


def multi_angle_between(a, b, n):
    """Calculates the angle between a and b around n

    Params:
        a: [n, x, y, z]
        b: [x, y, z]
        n: [x, y, z] axis around which to rotate
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.asarray(n)

    a_norm = a / np.linalg.norm(a, axis=1)[:, None]
    b_norm = b / np.linalg.norm(b)
    n_norm = n / np.linalg.norm(n)

    a_proj = a_norm - np.inner(a_norm, n_norm[None, :]) * n_norm
    b_proj = b_norm - np.inner(b_norm, n_norm) * n_norm

    a_proj /= np.linalg.norm(a_proj, axis=1)[:, None]
    b_proj /= np.linalg.norm(b_proj)

    return np.arccos(np.inner(a_proj, b_proj))


def angle_between(a, b, n):
    """Calculates the angle between a and b around n

    Params:
        a: [x, y, z]
        b: [x, y, z]
        n: [x, y, z] axis around which to rotate
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.asarray(n)

    a_norm = a / np.linalg.norm(a, axis=-1)
    b_norm = b / np.linalg.norm(b, axis=-1)
    n_norm = n / np.linalg.norm(n, axis=-1)

    a_proj = a_norm - np.inner(a_norm, n_norm) * n_norm
    b_proj = b_norm - np.inner(b_norm, n_norm) * n_norm

    a_proj /= np.linalg.norm(a_proj, axis=-1)
    b_proj /= np.linalg.norm(b_proj, axis=-1)

    return np.arccos(np.inner(a_proj, b_proj))
