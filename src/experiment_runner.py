import numpy as np
import json
import os
from src.tsp_utils import generate_tsp_instance, calculate_distance_matrix
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch


def run_experiments():
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)

    results = {"sa_configs": [], "hsa_configs": []}

    # SA Configurations (8)
    sa_params_list = [
        {"initial_temp": 1000, "cooling_rate": 0.99, "max_iter": 10000},
        {"initial_temp": 1000, "cooling_rate": 0.995, "max_iter": 10000},
        {"initial_temp": 1000, "cooling_rate": 0.999, "max_iter": 10000},
        {"initial_temp": 5000, "cooling_rate": 0.99, "max_iter": 10000},
        {"initial_temp": 5000, "cooling_rate": 0.995, "max_iter": 10000},
        {"initial_temp": 5000, "cooling_rate": 0.999, "max_iter": 10000},
        {"initial_temp": 1000, "cooling_rate": 0.999, "max_iter": 20000},
        {"initial_temp": 5000, "cooling_rate": 0.999, "max_iter": 20000},
    ]

    print("Running SA Experiments (5 trials per config)...")
    for i, params in enumerate(sa_params_list):
        print(f"Config SA-{i + 1}: {params}")
        trials = []
        for t in range(5):
            sa = SimulatedAnnealing(dist_matrix, **params)
            best_path, best_dist, history, exec_time = sa.solve()
            trials.append(
                {
                    "trial_id": t + 1,
                    "best_dist": float(best_dist),
                    "exec_time": float(exec_time),
                }
            )
        results["sa_configs"].append(
            {"config_id": f"SA-{i + 1}", "params": params, "trials": trials}
        )

    # HSA Configurations (8)
    hsa_params_list = [
        {"hms": 10, "hmcr": 0.8, "par": 0.2, "max_iter": 5000},
        {"hms": 10, "hmcr": 0.9, "par": 0.3, "max_iter": 5000},
        {"hms": 20, "hmcr": 0.8, "par": 0.2, "max_iter": 5000},
        {"hms": 20, "hmcr": 0.9, "par": 0.3, "max_iter": 5000},
        {"hms": 30, "hmcr": 0.9, "par": 0.3, "max_iter": 5000},
        {"hms": 20, "hmcr": 0.95, "par": 0.3, "max_iter": 5000},
        {"hms": 20, "hmcr": 0.9, "par": 0.4, "max_iter": 5000},
        {"hms": 20, "hmcr": 0.95, "par": 0.3, "max_iter": 10000},
    ]

    print("Running HSA Experiments (5 trials per config)...")
    for i, params in enumerate(hsa_params_list):
        print(f"Config HSA-{i + 1}: {params}")
        trials = []
        for t in range(5):
            hsa = HarmonySearch(dist_matrix, **params)
            best_path, best_dist, history, exec_time = hsa.solve()
            trials.append(
                {
                    "trial_id": t + 1,
                    "best_dist": float(best_dist),
                    "exec_time": float(exec_time),
                }
            )
        results["hsa_configs"].append(
            {"config_id": f"HSA-{i + 1}", "params": params, "trials": trials}
        )

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Experiments completed. Results saved to experiment_results.json")


if __name__ == "__main__":
    run_experiments()
