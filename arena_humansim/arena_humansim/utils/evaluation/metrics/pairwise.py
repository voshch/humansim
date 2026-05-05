import numpy as np
from scipy.spatial.distance import directed_hausdorff


def pairwise_hausdorff(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    return float(max(directed_hausdorff(traj_a, traj_b)[0], directed_hausdorff(traj_b, traj_a)[0]))
