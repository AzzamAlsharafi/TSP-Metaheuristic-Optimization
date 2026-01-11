import numpy as np
from src.tsp_utils import calculate_total_distance

def estimate_initial_temperature(dist_matrix, num_samples=100, p0=0.85, seed=None):
    if not (0 < p0 < 1):
        raise ValueError("p0 must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(dist_matrix)
    current = rng.permutation(n)
    deltas = []

    current_cost = calculate_total_distance(current, dist_matrix)

    for _ in range(num_samples):
        i, j = sorted(rng.choice(n, 2, replace=False))
        new = current.copy()
        new[i:j+1] = new[i:j+1][::-1]  # 2-opt reversal (inclusive j)

        new_cost = calculate_total_distance(new, dist_matrix)
        delta = new_cost - current_cost

        if delta > 0:
            deltas.append(delta)

        # walk forward
        current = new
        current_cost = new_cost

    if not deltas:
        return 1.0  # fallback

    delta_avg = float(np.mean(deltas))
    return -delta_avg / np.log(p0)
