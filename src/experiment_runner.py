import numpy as np
import json
import os
from src.tsp_utils import generate_tsp_instance, calculate_distance_matrix
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch
from intial_temp import estimate_initial_temperature


def calculate_initial_temperatures(dist_matrix):
    """Calculate T0 for p0=[0.85, 0.90, 0.95]"""
    print("="*60)
    print("PHASE 1: INITIAL TEMPERATURE ESTIMATION")
    print("="*60)
    print("\nCalculating scientifically-derived initial temperatures...")
    print("Method: T0 = -delta_avg / ln(p0)")
    print("  - Sampling: 100 random 2-opt moves")
    print("  - Seed: 999 (for reproducibility)\n")

    acceptance_probs = [0.85, 0.90, 0.95]
    temperatures = {}

    for p0 in acceptance_probs:
        T0 = estimate_initial_temperature(
            dist_matrix,
            num_samples=100,
            p0=p0,
            seed=999
        )
        temperatures[p0] = T0
        print(f"  p0 = {p0:.2f} → T0 = {T0:.4f}")

    print("\n" + "="*60)
    print("Temperature estimation complete!")
    print("="*60 + "\n")

    return temperatures


def run_experiments():
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)

    results = {"sa_configs": [], "hsa_configs": []}

    # Calculate initial temperatures
    initial_temps = calculate_initial_temperatures(dist_matrix)

    # SA Configurations (9 total: 3 temps × 3 cooling_rates)
    sa_params_list = []
    temp_values = [initial_temps[0.85], initial_temps[0.90], initial_temps[0.95]]
    cooling_rates = [0.90, 0.95, 0.99]

    for temp in temp_values:
        for cooling_rate in cooling_rates:
            sa_params_list.append({
                "initial_temp": temp,
                "cooling_rate": cooling_rate,
                "max_iter": 10000
            })

    print(f"Generated {len(sa_params_list)} SA configurations")
    print("Running SA Experiments (9 configs × 5 trials = 45 trials)...")
    for i, params in enumerate(sa_params_list):
        print(f"Config SA-{i + 1}: {params}")
        trials = []
        for t in range(5):
            # Use different random seed for each trial for reproducibility
            random_seed = 42 + i * 5 + t  # Unique seed per config and trial
            sa = SimulatedAnnealing(dist_matrix, random_seed=random_seed, **params)
            best_path, best_dist, history, exec_time, _ = sa.solve()  # Ignore metadata
            trials.append(
                {
                    "trial_id": t + 1,
                    "best_dist": float(best_dist),
                    "exec_time": float(exec_time),
                    "random_seed": random_seed,
                }
            )
        results["sa_configs"].append(
            {"config_id": f"SA-{i + 1}", "params": params, "trials": trials}
        )

    # HSA Configurations (9 total: 3 HMS × 1 HMCR × 3 PAR)
    hsa_params_list = []
    hms_values = [30, 60, 90]
    hmcr_values = [0.3]
    par_values = [0.3, 0.6, 0.9]

    for hms in hms_values:
        for hmcr in hmcr_values:
            for par in par_values:
                hsa_params_list.append({
                    "hms": hms,
                    "hmcr": hmcr,
                    "par": par,
                    "max_iter": 10000
                })

    print(f"Generated {len(hsa_params_list)} HSA configurations")
    print("Running HSA Experiments (9 configs × 5 trials = 45 trials)...")
    for i, params in enumerate(hsa_params_list):
        print(f"Config HSA-{i + 1}: {params}")
        trials = []
        for t in range(5):
            # Use different random seed for each trial for reproducibility
            random_seed = 1000 + i * 5 + t  # Unique seed per config and trial
            hsa = HarmonySearch(dist_matrix, random_seed=random_seed, **params)
            best_path, best_dist, history, exec_time = hsa.solve()
            trials.append(
                {
                    "trial_id": t + 1,
                    "best_dist": float(best_dist),
                    "exec_time": float(exec_time),
                    "random_seed": random_seed,
                }
            )
        results["hsa_configs"].append(
            {"config_id": f"HSA-{i + 1}", "params": params, "trials": trials}
        )

    # Add temperature metadata to results
    output = {
        "metadata": {
            "total_trials": len(sa_params_list) * 5 + len(hsa_params_list) * 5,
            "sa_configs_count": len(sa_params_list),
            "hsa_configs_count": len(hsa_params_list),
            "initial_temperatures": {
                "p0_0.85": initial_temps[0.85],
                "p0_0.90": initial_temps[0.90],
                "p0_0.95": initial_temps[0.95]
            },
            "temp_estimation_seed": 999,
            "temp_estimation_samples": 100
        },
        "sa_configs": results["sa_configs"],
        "hsa_configs": results["hsa_configs"]
    }

    with open("experiment_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nExperiments completed! Results saved to experiment_results.json")
    print(f"  Total trials: {len(sa_params_list) * 5 + len(hsa_params_list) * 5}")
    print(f"  SA trials: {len(sa_params_list) * 5}")
    print(f"  HSA trials: {len(hsa_params_list) * 5}")


if __name__ == "__main__":
    run_experiments()
