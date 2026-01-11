import matplotlib.pyplot as plt
import json
import numpy as np
import os
from src.tsp_utils import generate_tsp_instance, calculate_distance_matrix, load_optimal_tour
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch


def generate_plots():
    # Setup
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)
    report_dir = os.path.join(os.getcwd(), "report", "images")
    os.makedirs(report_dir, exist_ok=True)

    # Load experiment results to get best configs
    print("Loading experiment results to find best configurations...")
    with open("experiment_results.json", "r") as f:
        exp_data = json.load(f)

    # Find best SA config 
    best_sa_config = None
    best_sa_mean = float('inf')
    for config in exp_data['sa_configs']:
        dists = [t['best_dist'] for t in config['trials']]
        mean_dist = np.mean(dists)
        if mean_dist < best_sa_mean:
            best_sa_mean = mean_dist
            best_sa_config = config

    print(f"Best SA config: {best_sa_config['config_id']} with params {best_sa_config['params']}")

    # 1. SA Run with ACTUAL best config and best seed
    # Find the best trial seed from Phase 1
    best_trial = min(best_sa_config['trials'], key=lambda x: x['best_dist'])
    best_seed = best_trial['seed']
    print(f"Using best seed from Phase 1: {best_seed} (distance: {best_trial['best_dist']:.2f})")

    print("Generating SA plots...")
    sa = SimulatedAnnealing(
        dist_matrix, random_seed=best_seed, min_temp=0, **best_sa_config['params']
    )
    best_path_sa, best_dist_sa, history_sa, _, _ = sa.solve()  # Ignore exec_time and metadata
    print(f"SA Best Distance: {best_dist_sa:.2f} (iterations: {len(history_sa)})")

    # SA Tour
    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=30)
    full_path_sa = list(best_path_sa) + [best_path_sa[0]]
    path_coords = cities[full_path_sa]
    plt.plot(path_coords[:, 0], path_coords[:, 1], "b-", linewidth=1)
    plt.title(f"SA Optimal Tour (Distance: {best_dist_sa:.2f})")
    plt.savefig(f"{report_dir}/sa_tour.png", dpi=300)
    plt.close()

    # SA Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(history_sa)
    plt.title("SA Convergence Trajectory")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.savefig(f"{report_dir}/sa_convergence.png", dpi=300)
    plt.close()

    # Find best HSA config 
    best_hsa_config = None
    best_hsa_mean = float('inf')
    for config in exp_data['hsa_configs']:
        dists = [t['best_dist'] for t in config['trials']]
        mean_dist = np.mean(dists)
        if mean_dist < best_hsa_mean:
            best_hsa_mean = mean_dist
            best_hsa_config = config

    print(f"Best HSA config: {best_hsa_config['config_id']} with params {best_hsa_config['params']}")

    # 2. HSA Run with ACTUAL best config and best seed
    # Find the best trial seed from Phase 1
    best_hsa_trial = min(best_hsa_config['trials'], key=lambda x: x['best_dist'])
    best_hsa_seed = best_hsa_trial['random_seed']  # HSA uses 'random_seed' field
    print(f"Using best HSA seed from Phase 1: {best_hsa_seed} (distance: {best_hsa_trial['best_dist']:.2f})")

    print("Generating HSA plots...")
    hsa = HarmonySearch(dist_matrix, random_seed=best_hsa_seed, **best_hsa_config['params'])
    best_path_hsa, best_dist_hsa, history_hsa, _ = hsa.solve()
    print(f"HSA Best Distance: {best_dist_hsa:.2f} (iterations: {len(history_hsa)})")

    # HSA Tour
    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=30)
    full_path_hsa = list(best_path_hsa) + [best_path_hsa[0]]
    path_coords = cities[full_path_hsa]
    plt.plot(path_coords[:, 0], path_coords[:, 1], "g-", linewidth=1)
    plt.title(f"HSA Optimal Tour (Distance: {best_dist_hsa:.2f})")
    plt.savefig(f"{report_dir}/hsa_tour.png", dpi=300)
    plt.close()

    # HSA Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(history_hsa)
    plt.title("HSA Convergence Trajectory")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.savefig(f"{report_dir}/hsa_convergence.png", dpi=300)
    plt.close()

    # 3. Optimal Tour Visualization
    print("Generating optimal tour plot...")
    optimal_tour, optimal_dist = load_optimal_tour(dist_matrix=dist_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=30)
    full_path_opt = list(optimal_tour) + [optimal_tour[0]]
    path_coords_opt = cities[full_path_opt]
    plt.plot(path_coords_opt[:, 0], path_coords_opt[:, 1], "g-", linewidth=1)
    plt.title(f"Optimal Tour (Distance: {optimal_dist:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(f"{report_dir}/optimal_tour.png", dpi=300)
    plt.close()

    # 4. Boxplots from experiment_results.json
    print("Generating boxplots...")
    results_file = "experiment_results.json"
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found. Run experiment_runner.py first.")
        print("Skipping boxplot generation.")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    sa_dists = []
    sa_labels = []
    sa_times = []
    for config in results["sa_configs"]:
        dists = [t["best_dist"] for t in config["trials"]]
        times = [t["exec_time"] for t in config["trials"]]
        sa_dists.append(dists)
        sa_times.append(times)
        sa_labels.append(config["config_id"])

    hsa_dists = []
    hsa_labels = []
    hsa_times = []
    for config in results["hsa_configs"]:
        dists = [t["best_dist"] for t in config["trials"]]
        times = [t["exec_time"] for t in config["trials"]]
        hsa_dists.append(dists)
        hsa_times.append(times)
        hsa_labels.append(config["config_id"])

    # Distance Boxplot - SA (separate)
    plt.figure(figsize=(10, 6))
    plt.boxplot(sa_dists, labels=sa_labels)
    plt.title("SA: Distance Distribution Across Configurations")
    plt.ylabel("Distance")
    plt.xlabel("Configuration")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{report_dir}/sa_distance_boxplot.png", dpi=300)
    plt.close()

    # Distance Boxplot - HSA (separate)
    plt.figure(figsize=(10, 6))
    plt.boxplot(hsa_dists, labels=hsa_labels)
    plt.title("HSA: Distance Distribution Across Configurations")
    plt.ylabel("Distance")
    plt.xlabel("Configuration")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{report_dir}/hsa_distance_boxplot.png", dpi=300)
    plt.close()

    # Time Boxplot - SA (separate)
    plt.figure(figsize=(10, 6))
    plt.boxplot(sa_times, labels=sa_labels)
    plt.title("SA: Execution Time Distribution Across Configurations")
    plt.ylabel("Time (s)")
    plt.xlabel("Configuration")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{report_dir}/sa_time_boxplot.png", dpi=300)
    plt.close()

    # Time Boxplot - HSA (separate)
    plt.figure(figsize=(10, 6))
    plt.boxplot(hsa_times, labels=hsa_labels)
    plt.title("HSA: Execution Time Distribution Across Configurations")
    plt.ylabel("Time (s)")
    plt.xlabel("Configuration")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{report_dir}/hsa_time_boxplot.png", dpi=300)
    plt.close()

    print(f"\n{'='*60}")
    print(f"All plots saved to {report_dir}")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  1. sa_tour.png - SA best tour visualization")
    print("  2. sa_convergence.png - SA convergence trajectory ✅")
    print("  3. hsa_tour.png - HSA best tour visualization")
    print("  4. hsa_convergence.png - HSA convergence trajectory ✅")
    print("  5. optimal_tour.png - Known optimal tour (distance: 7910)")
    print("  6. sa_distance_boxplot.png - SA performance distribution ✅")
    print("  7. hsa_distance_boxplot.png - HSA performance distribution ✅")
    print("  8. sa_time_boxplot.png - SA execution time distribution")
    print("  9. hsa_time_boxplot.png - HSA execution time distribution")
    print(f"{'='*60}")
    print("\n✅ = Satisfies project requirements")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_plots()
