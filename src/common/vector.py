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

    print(a_norm, b_norm)
    print(np.dot(a, b))

    return np.arccos(np.dot(a_norm, b_norm))

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

    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    n_norm = n / np.linalg.norm(n)

    a_proj = a_norm - np.dot(a_norm, n_norm) * n_norm
    b_proj = b_norm - np.dot(b_norm, n_norm) * n_norm

    a_proj /= np.linalg.norm(a_proj)
    b_proj /= np.linalg.norm(b_proj)

    return np.arccos(np.dot(a_proj, b_proj))
