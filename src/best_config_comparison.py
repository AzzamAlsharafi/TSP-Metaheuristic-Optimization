import numpy as np
import json
import os
from scipy import stats
from src.tsp_utils import generate_tsp_instance, calculate_distance_matrix
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch


def find_best_configs(results):
    """
    Analyzes experiment_results.json to find best SA and HSA configurations.
    Returns the config with lowest mean distance for each algorithm.
    """
    best_sa_config = None
    best_sa_mean = float('inf')

    # Find best SA config
    for config in results['sa_configs']:
        distances = [trial['best_dist'] for trial in config['trials']]
        mean_dist = np.mean(distances)

        if mean_dist < best_sa_mean:
            best_sa_mean = mean_dist
            best_sa_config = config

    best_hsa_config = None
    best_hsa_mean = float('inf')

    # Find best HSA config
    for config in results['hsa_configs']:
        distances = [trial['best_dist'] for trial in config['trials']]
        mean_dist = np.mean(distances)

        if mean_dist < best_hsa_mean:
            best_hsa_mean = mean_dist
            best_hsa_config = config

    return best_sa_config, best_hsa_config


def run_comparison_trials(n_trials=30):
    """
    Phase 2: Statistical Validation

    Runs the best SA and HSA configurations 30 times each for robust
    statistical comparison with predetermined seeds for reproducibility.
    """
    # Check if Phase 1 results exist
    if not os.path.exists("experiment_results.json"):
        print("ERROR: experiment_results.json not found!")
        print("Please run 'PYTHONPATH=. python3 src/experiment_runner.py' first (Phase 1)")
        return

    # Load Phase 1 results
    with open("experiment_results.json", "r") as f:
        phase1_results = json.load(f)

    # Find best configurations
    print("Analyzing Phase 1 results to find best configurations...")
    best_sa_config, best_hsa_config = find_best_configs(phase1_results)

    print(f"\nBest SA Config: {best_sa_config['config_id']}")
    print(f"  Parameters: {best_sa_config['params']}")
    print(f"  Phase 1 mean: {np.mean([t['best_dist'] for t in best_sa_config['trials']]):.2f}")

    print(f"\nBest HSA Config: {best_hsa_config['config_id']}")
    print(f"  Parameters: {best_hsa_config['params']}")
    print(f"  Phase 1 mean: {np.mean([t['best_dist'] for t in best_hsa_config['trials']]):.2f}")

    # Load TSP instance
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)

    # Phase 2: Run best SA config 30 times
    print(f"\n{'='*60}")
    print(f"Phase 2: Running best SA config {n_trials} times...")
    print(f"{'='*60}")

    sa_results = []
    sa_params = best_sa_config['params']

    for trial in range(n_trials):
        # Predetermined seeds for reproducibility (Option A)
        random_seed = 5000 + trial  # Seeds: 5000, 5001, 5002, ..., 5029
        print(f"  SA Trial {trial+1}/{n_trials}...", end='\r')

        sa = SimulatedAnnealing(dist_matrix, random_seed=random_seed, **sa_params)
        best_path, best_dist, history, exec_time = sa.solve()

        sa_results.append({
            'trial_id': trial + 1,
            'best_dist': float(best_dist),
            'exec_time': float(exec_time),
            'random_seed': random_seed
        })

    print(f"  SA completed: {n_trials} trials                    ")

    # Phase 2: Run best HSA config 30 times
    print(f"\n{'='*60}")
    print(f"Phase 2: Running best HSA config {n_trials} times...")
    print(f"{'='*60}")

    hsa_results = []
    hsa_params = best_hsa_config['params']

    for trial in range(n_trials):
        # Predetermined seeds for reproducibility (Option A)
        random_seed = 6000 + trial  # Seeds: 6000, 6001, 6002, ..., 6029
        print(f"  HSA Trial {trial+1}/{n_trials}...", end='\r')

        hsa = HarmonySearch(dist_matrix, random_seed=random_seed, **hsa_params)
        best_path, best_dist, history, exec_time = hsa.solve()

        hsa_results.append({
            'trial_id': trial + 1,
            'best_dist': float(best_dist),
            'exec_time': float(exec_time),
            'random_seed': random_seed
        })

    print(f"  HSA completed: {n_trials} trials                    ")

    # Statistical Analysis
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    sa_distances = [r['best_dist'] for r in sa_results]
    hsa_distances = [r['best_dist'] for r in hsa_results]

    sa_times = [r['exec_time'] for r in sa_results]
    hsa_times = [r['exec_time'] for r in hsa_results]

    # Distance statistics
    sa_mean = np.mean(sa_distances)
    sa_std = np.std(sa_distances, ddof=1)
    sa_min = np.min(sa_distances)
    sa_max = np.max(sa_distances)

    hsa_mean = np.mean(hsa_distances)
    hsa_std = np.std(hsa_distances, ddof=1)
    hsa_min = np.min(hsa_distances)
    hsa_max = np.max(hsa_distances)

    # Time statistics
    sa_time_mean = np.mean(sa_times)
    hsa_time_mean = np.mean(hsa_times)

    print(f"\nSA ({best_sa_config['config_id']}) - Distance:")
    print(f"  Mean:   {sa_mean:.2f}")
    print(f"  Std:    {sa_std:.2f}")
    print(f"  Min:    {sa_min:.2f}")
    print(f"  Max:    {sa_max:.2f}")
    print(f"  95% CI: [{sa_mean - 1.96*sa_std/np.sqrt(n_trials):.2f}, "
          f"{sa_mean + 1.96*sa_std/np.sqrt(n_trials):.2f}]")
    print(f"  Avg Time: {sa_time_mean:.2f}s")

    print(f"\nHSA ({best_hsa_config['config_id']}) - Distance:")
    print(f"  Mean:   {hsa_mean:.2f}")
    print(f"  Std:    {hsa_std:.2f}")
    print(f"  Min:    {hsa_min:.2f}")
    print(f"  Max:    {hsa_max:.2f}")
    print(f"  95% CI: [{hsa_mean - 1.96*hsa_std/np.sqrt(n_trials):.2f}, "
          f"{hsa_mean + 1.96*hsa_std/np.sqrt(n_trials):.2f}]")
    print(f"  Avg Time: {hsa_time_mean:.2f}s")

    # Statistical tests
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*60}")

    # Independent t-test for distance
    t_stat, p_value = stats.ttest_ind(sa_distances, hsa_distances)

    print(f"\nIndependent t-test (Distance):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")

    if p_value < 0.05:
        winner = "SA" if sa_mean < hsa_mean else "HSA"
        print(f"  Result: {winner} is SIGNIFICANTLY better (p < 0.05) ✓")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_trials-1)*sa_std**2 + (n_trials-1)*hsa_std**2) / (2*n_trials-2))
    cohens_d = abs(sa_mean - hsa_mean) / pooled_std

    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if cohens_d < 0.2:
        print(f"  Interpretation: Small effect")
    elif cohens_d < 0.5:
        print(f"  Interpretation: Medium effect")
    else:
        print(f"  Interpretation: Large effect")

    # Percentage difference
    pct_diff = abs(sa_mean - hsa_mean) / max(sa_mean, hsa_mean) * 100
    better_algo = "SA" if sa_mean < hsa_mean else "HSA"
    print(f"\nPercentage Difference: {pct_diff:.2f}%")
    print(f"  {better_algo} found solutions {pct_diff:.2f}% better on average")

    # Save results
    results = {
        'phase': 'Phase 2 - Best Config Comparison',
        'n_trials': n_trials,
        'seed_range': {
            'sa': '5000-5029',
            'hsa': '6000-6029'
        },
        'sa': {
            'config_id': best_sa_config['config_id'],
            'params': sa_params,
            'trials': sa_results,
            'statistics': {
                'mean': float(sa_mean),
                'std': float(sa_std),
                'min': float(sa_min),
                'max': float(sa_max),
                'mean_time': float(sa_time_mean),
                'ci_95': [
                    float(sa_mean - 1.96*sa_std/np.sqrt(n_trials)),
                    float(sa_mean + 1.96*sa_std/np.sqrt(n_trials))
                ]
            }
        },
        'hsa': {
            'config_id': best_hsa_config['config_id'],
            'params': hsa_params,
            'trials': hsa_results,
            'statistics': {
                'mean': float(hsa_mean),
                'std': float(hsa_std),
                'min': float(hsa_min),
                'max': float(hsa_max),
                'mean_time': float(hsa_time_mean),
                'ci_95': [
                    float(hsa_mean - 1.96*hsa_std/np.sqrt(n_trials)),
                    float(hsa_mean + 1.96*hsa_std/np.sqrt(n_trials))
                ]
            }
        },
        'statistical_tests': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'percentage_difference': float(pct_diff),
            'better_algorithm': better_algo
        }
    }

    with open("best_config_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Results saved to: best_config_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    print(f"{'='*60}")
    print("PHASE 2: BEST CONFIGURATION COMPARISON")
    print(f"{'='*60}")
    print("\nThis script runs the best SA and HSA configurations")
    print("30 times each for robust statistical comparison.")
    print("\nPredetermined seeds ensure reproducibility:")
    print("  - SA:  seeds 5000-5029")
    print("  - HSA: seeds 6000-6029\n")

    run_comparison_trials(n_trials=30)
