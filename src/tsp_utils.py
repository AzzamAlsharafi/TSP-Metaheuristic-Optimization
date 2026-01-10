import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import tsplib95

# Known optimal distance for rd100 benchmark
RD100_OPTIMAL_DISTANCE = 7910


def load_tsplib_instance(tsp_file="data/rd100.tsp"):
    """
    Loads a TSP instance from TSPLIB format.

    Args:
        tsp_file: Path to .tsp file

    Returns:
        cities: numpy array of shape (n_cities, 2) with 0-based indexing
        metadata: dict with name, dimension, comment
    """
    problem = tsplib95.load(tsp_file)

    # Extract coordinates (TSPLIB uses 1-based indexing)
    coords = []
    for node in problem.get_nodes():
        x, y = problem.node_coords[node]
        coords.append([x, y])

    cities = np.array(coords)

    metadata = {
        'name': problem.name,
        'dimension': problem.dimension,
        'comment': getattr(problem, 'comment', ''),
    }

    return cities, metadata


def load_optimal_tour(tour_file="data/rd100.opt.tour", dist_matrix=None):
    """
    Loads the known optimal tour from TSPLIB format.

    Args:
        tour_file: Path to .opt.tour file
        dist_matrix: Optional distance matrix to verify tour distance

    Returns:
        tour: numpy array with 0-based city indices
        optimal_distance: known optimal distance (7910 for rd100)
    """
    problem = tsplib95.load(tour_file)

    # Convert 1-based to 0-based indexing
    tour = np.array([node - 1 for node in problem.tours[0]])

    optimal_distance = RD100_OPTIMAL_DISTANCE

    # Verify distance if matrix provided
    if dist_matrix is not None:
        calculated = calculate_total_distance(tour, dist_matrix)
        print(f"Optimal tour distance: {calculated:.2f} (known: {optimal_distance})")

    return tour, optimal_distance


def generate_tsp_instance(n_cities=100, seed=42, tsp_file="data/rd100.tsp"):
    """
    Loads TSP instance from TSPLIB format (rd100 benchmark).

    For backward compatibility, still accepts n_cities and seed params but
    ignores them when loading from TSPLIB file.

    Args:
        n_cities: Number of cities (ignored when loading TSPLIB)
        seed: Random seed (ignored when loading TSPLIB)
        tsp_file: Path to TSPLIB .tsp file (default: data/rd100.tsp)

    Returns:
        cities: numpy array of shape (n_cities, 2) with city coordinates
    """
    if os.path.exists(tsp_file):
        print(f"Loading TSPLIB instance from {tsp_file}")
        cities, metadata = load_tsplib_instance(tsp_file)
        print(f"Loaded {metadata['name']}: {len(cities)} cities")
        return cities
    else:
        raise FileNotFoundError(f"TSPLIB file not found: {tsp_file}")


def calculate_distance_matrix(cities):
    """
    Calculates the Euclidean distance matrix between all pairs of cities.
    """
    n_cities = len(cities)
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def calculate_total_distance(path, dist_matrix):
    """
    Calculates the total distance of a given path (tour).
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_matrix[path[i], path[i+1]]
    distance += dist_matrix[path[-1], path[0]]  # Return to start
    return distance


def get_optimal_distance():
    """
    Returns the known optimal distance for the rd100 benchmark.
    """
    return RD100_OPTIMAL_DISTANCE


def plot_tsp_instance(cities, title="TSP Instance"):
    """
    Plots the cities as a graph.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', edgecolors='k')
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_path = "tsp_instance.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_tsp_solution(cities, path, distance, title="TSP Solution"):
    """
    Plots the TSP solution (tour).
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', edgecolors='k')

    # Draw the path
    for i in range(len(path) - 1):
        plt.plot([cities[path[i], 0], cities[path[i+1], 0]],
                 [cities[path[i], 1], cities[path[i+1], 1]], 'b-', alpha=0.6)
    # Close the loop
    plt.plot([cities[path[-1], 0], cities[path[0], 0]],
             [cities[path[-1], 1], cities[path[0], 1]], 'b-', alpha=0.6)

    plt.title(f"{title}\nTotal Distance: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_path = "tsp_solution.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


if __name__ == "__main__":
    # Load and visualize the rd100 instance
    cities = generate_tsp_instance()
    print(f"Loaded TSP instance with {len(cities)} cities.")

    dist_matrix = calculate_distance_matrix(cities)

    # Load and verify optimal tour
    optimal_tour, optimal_dist = load_optimal_tour(dist_matrix=dist_matrix)
    print(f"Known optimal distance: {optimal_dist}")

    plot_path = plot_tsp_instance(cities, title="rd100 TSP Instance")
    print(f"Instance visualization saved to {plot_path}")
