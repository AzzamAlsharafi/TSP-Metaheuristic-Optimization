import numpy as np
import matplotlib.pyplot as plt
import json
import os

def generate_tsp_instance(n_cities=100, seed=42):
    """
    Generates a hardcoded TSP instance with n_cities.
    Uses a fixed seed for reproducibility.
    """
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2) * 1000  # Scale to 1000x1000 area
    return cities

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
    # Generate and visualize the instance
    cities = generate_tsp_instance(100)
    print(f"Generated TSP instance with {len(cities)} cities.")
    
    # Save cities to a file for consistency across algorithms
    np.save("cities.npy", cities)
    
    plot_path = plot_tsp_instance(cities)
    print(f"Instance visualization saved to {plot_path}")
