import numpy as np
import time
from src.tsp_utils import calculate_total_distance


class SimulatedAnnealing:
    def __init__(
        self,
        dist_matrix,
        initial_temp=1000,
        cooling_rate=0.995,
        min_temp=0,
        max_iter=10000,
        random_seed=None,
    ):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter
        self.random_seed = random_seed

    def get_neighbor(self, path):
        """
        Generates a neighbor using proper 2-opt move.

        2-opt: Removes two edges and reconnects by reversing the segment between them.
        Example: [0,1,2,3,4,5] with i=1, j=4 becomes [0,1,4,3,2,5]
                 (reverses segment from index i+1 to j)
        """
        neighbor = path.copy()
        # Select two different positions, ensure i < j
        i, j = sorted(np.random.choice(len(path), 2, replace=False))

        # Reverse the segment between i and j (inclusive of both endpoints)
        neighbor[i:j+1] = neighbor[i:j+1][::-1]

        return neighbor

    def solve(self):
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Initial solution
        current_path = np.random.permutation(self.n_cities)
        current_dist = calculate_total_distance(current_path, self.dist_matrix)

        best_path = current_path.copy()
        best_dist = current_dist

        temp = self.initial_temp
        history = [current_dist]

        start_time = time.time()
        early_stopped = False
        iterations_completed = 0

        for i in range(self.max_iter):
            if temp < self.min_temp:
                early_stopped = True
                iterations_completed = i
                break

            neighbor = self.get_neighbor(current_path)
            neighbor_dist = calculate_total_distance(neighbor, self.dist_matrix)

            delta = neighbor_dist - current_dist

            # Acceptance criteria
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_path = neighbor
                current_dist = neighbor_dist

                if current_dist < best_dist:
                    best_path = current_path.copy()
                    best_dist = current_dist

            history.append(best_dist)
            temp *= self.cooling_rate
            iterations_completed = i + 1

        execution_time = time.time() - start_time

        # Return metadata about the run
        metadata = {
            'early_stopped': early_stopped,
            'iterations_completed': iterations_completed,
            'final_temperature': temp
        }

        return best_path, best_dist, history, execution_time, metadata


if __name__ == "__main__":
    from src.tsp_utils import (
        generate_tsp_instance,
        calculate_distance_matrix,
        plot_tsp_solution,
    )

    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)

    sa = SimulatedAnnealing(
        dist_matrix, initial_temp=1000, cooling_rate=0.999, max_iter=50000
    )
    best_path, best_dist, history, exec_time, metadata = sa.solve()

    print(f"SA Best Distance: {best_dist:.2f}")
    print(f"Execution Time: {exec_time:.2f}s")
    print(f"Iterations Completed: {metadata['iterations_completed']}")
    if metadata['early_stopped']:
        print(f"⚠️ Early Stopping: Stopped at iteration {metadata['iterations_completed']} (temp: {metadata['final_temperature']:.6f})")
    else:
        print(f"✓ No Early Stopping: Completed full {metadata['iterations_completed']} iterations")
    print(f"Final Temperature: {metadata['final_temperature']:.6f}")

    plot_tsp_solution(cities, best_path, best_dist, title="SA Solution")
