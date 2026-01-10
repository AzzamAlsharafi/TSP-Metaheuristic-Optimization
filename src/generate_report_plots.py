import matplotlib.pyplot as plt
import json
import numpy as np
import os
from src.tsp_utils import generate_tsp_instance, calculate_distance_matrix
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch


def generate_plots():
    # Setup
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)
    report_dir = os.path.join(os.getcwd(), "report", "images")
    os.makedirs(report_dir, exist_ok=True)

    # 1. SA Run
    print("Generating SA plots...")
    sa = SimulatedAnnealing(
        dist_matrix, initial_temp=1000, cooling_rate=0.999, max_iter=20000, random_seed=42
    )
    best_path_sa, best_dist_sa, history_sa, _ = sa.solve()

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

    # 2. HSA Run
    print("Generating HSA plots...")
    hsa = HarmonySearch(dist_matrix, hms=20, hmcr=0.95, par=0.3, max_iter=10000, random_seed=1000)
    best_path_hsa, best_dist_hsa, history_hsa, _ = hsa.solve()

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

    # 3. Boxplots from experiment_results.json
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

    # Distance Boxplot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(sa_dists, labels=sa_labels)
    plt.title("SA Distance Distribution")
    plt.ylabel("Distance")

    plt.subplot(1, 2, 2)
    plt.boxplot(hsa_dists, labels=hsa_labels)
    plt.title("HSA Distance Distribution")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/distance_boxplots.png", dpi=300)
    plt.close()

    # Time Boxplot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(sa_times, labels=sa_labels)
    plt.title("SA Execution Time")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.boxplot(hsa_times, labels=hsa_labels)
    plt.title("HSA Execution Time")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/time_boxplots.png", dpi=300)
    plt.close()

    print(f"All plots saved to {report_dir}")


if __name__ == "__main__":
    generate_plots()
