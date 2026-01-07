import numpy as np
import time
from src.tsp_utils import calculate_total_distance


class HarmonySearch:
    def __init__(self, dist_matrix, hms=20, hmcr=0.9, par=0.3, max_iter=10000):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.hms = hms  # Harmony Memory Size
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par  # Pitch Adjusting Rate
        self.max_iter = max_iter

    def initialize_hm(self):
        """
        Initializes the Harmony Memory with random tours.
        """
        hm = []
        for _ in range(self.hms):
            path = np.random.permutation(self.n_cities)
            dist = calculate_total_distance(path, self.dist_matrix)
            hm.append((path, dist))
        return hm

    def improvise(self, hm):
        """
        Improvises a new harmony (tour).
        """
        new_path = np.full(self.n_cities, -1, dtype=int)
        used_cities = set()

        for i in range(self.n_cities):
            if np.random.rand() < self.hmcr:
                # Memory consideration
                random_harmony_idx = np.random.randint(0, self.hms)
                city = hm[random_harmony_idx][0][i]

                if city not in used_cities:
                    new_path[i] = city
                    used_cities.add(city)
                else:
                    # City already used, pick a random available city
                    available = list(set(range(self.n_cities)) - used_cities)
                    city = np.random.choice(available)
                    new_path[i] = city
                    used_cities.add(city)
            else:
                # Random selection
                available = list(set(range(self.n_cities)) - used_cities)
                city = np.random.choice(available)
                new_path[i] = city
                used_cities.add(city)

            # Pitch adjustment (swap with a previous city in the new path)
            if np.random.rand() < self.par and i > 0:
                idx = np.random.randint(0, i)
                new_path[i], new_path[idx] = new_path[idx], new_path[i]

        return new_path

    def solve(self):
        hm = self.initialize_hm()
        hm.sort(key=lambda x: x[1])  # Sort by distance (best first)

        best_dist = hm[0][1]
        best_path = hm[0][0].copy()
        history = [best_dist]

        start_time = time.time()

        for _ in range(self.max_iter):
            new_path = self.improvise(hm)
            new_dist = calculate_total_distance(new_path, self.dist_matrix)

            # If new harmony is better than the worst in HM, replace it
            if new_dist < hm[-1][1]:
                hm[-1] = (new_path, new_dist)
                hm.sort(key=lambda x: x[1])

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_path = new_path.copy()

            history.append(best_dist)

        execution_time = time.time() - start_time

        return best_path, best_dist, history, execution_time


if __name__ == "__main__":
    from src.tsp_utils import (
        generate_tsp_instance,
        calculate_distance_matrix,
        plot_tsp_solution,
    )

    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)

    hsa = HarmonySearch(dist_matrix, hms=30, hmcr=0.95, par=0.4, max_iter=20000)
    best_path, best_dist, history, exec_time = hsa.solve()

    print(f"HSA Best Distance: {best_dist:.2f}")
    print(f"Execution Time: {exec_time:.2f}s")

    plot_tsp_solution(cities, best_path, best_dist, title="HSA Solution")
