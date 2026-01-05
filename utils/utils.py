import numpy as np

def rotate_to_fixed_frame(vectors, base_dir):
    """
    Rotate vectors to a fixed frame defined by base_dir (NumPy version)

    Args:
        vectors: numpy array of shape (N, 2)
        base_dir: numpy array of shape (2,)

    Returns:
        rotated_vectors: numpy array of shape (N, 2)
    """
    norm = np.linalg.norm(base_dir)
    if norm < 1e-12:
        raise ValueError("base_dir norm too small")

    base = base_dir / norm
    x_axis = base
    y_axis = np.array([-base[1], base[0]])

    R = np.stack([x_axis, y_axis], axis=1)  # (2,2)
    return vectors @ R

def estimate_base_angle(points_xy, n_segments=10):
    """
    Estimate the base angle of a sequence of 2D points.
    
    Args:
        points_xy: numpy array of shape (N, 2)
        n_segments: number of segments to consider for the estimation
    
    Returns:
        base_angle: numpy array of shape (2,), representing the average direction vector
    """
    if points_xy.shape[0] < 2:
        return 0.0
    
    m = int(min(max(1, n_segments), points_xy.shape[0] - 1))
    seg = points_xy[1:m+1] - points_xy[0]
    v = seg.mean(axis=0)

    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        v = points_xy[1] - points_xy[0]

    return v


def rot2(theta):
    """
    Create a 2D rotation matrix for angle theta.

    Args:
        theta: rotation angle in radians
    
    Returns:
        R: 2x2 rotation matrix as a numpy array
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]])

def wrap_pi(a):
    """
    Wrap angle to [-pi, pi].

    Args:
        a: angle in radians (numpy array or float)
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

def find_anchor_at_angle(points, angle_target=np.deg2rad(30), min_r=1e-6, window=10):
    """
    Find the index of the first point that crosses the target angle

    Args:
        points: list or numpy array of shape (N, 2)
        angle_target: target angle in radians
        min_r: minimum radius to consider
        window: number of points to use for estimating initial direction

    Returns:
        index of the anchor point, or None if not found
    """
    pts = np.asarray(points)

    if pts.shape[0] < 2:
        return None

    w = min(window, pts.shape[0] - 1)
    segs = pts[1:w+1] - pts[0]
    v0 = segs.mean(axis=0)
    if not np.isfinite(v0).all() or np.linalg.norm(v0) < 1e-12:
        v0 = pts[1] - pts[0]
    phi0 = np.arctan2(v0[1], v0[0])

    v = pts - pts[0]
    r = np.linalg.norm(v, axis=1)
    th = wrap_pi(np.arctan2(v[:, 1], v[:, 0]) - phi0)

    valid = r > min_r
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        return None

    # First crossing of +target or -target
    for i in idxs:
        if abs(abs(th[i]) - angle_target) < np.deg2rad(2):
            return i

    # Fallback: nearest
    return int(idxs[np.argmin(np.abs(np.abs(th[idxs]) - angle_target))])
