# TSP Metaheuristic Optimization

This project implements and compares two metaheuristic algorithms—**Simulated Annealing (SA)** and the **Harmony Search Algorithm (HSA)**—to solve the Traveling Salesman Problem (TSP). It includes a comprehensive experimental framework and an interactive Streamlit dashboard for visualization and analysis.

## Project Overview

The goal of this project is to find the shortest possible route that visits 100 cities exactly once and returns to the origin city. Since TSP is NP-hard, we use metaheuristics to find high-quality solutions efficiently.

### Implemented Algorithms
- **Simulated Annealing (SA)**: A trajectory-based metaheuristic inspired by the annealing process in metallurgy. It uses a 2-opt neighborhood structure and a geometric cooling schedule.
- **Harmony Search Algorithm (HSA)**: A population-based metaheuristic inspired by the improvisation process of musicians. It uses Harmony Memory (HM), Pitch Adjusting Rate (PAR), and Harmony Memory Consideration Rate (HMCR).

## Project Structure

- `src/`: Core implementation files.
    - `sa.py`: Simulated Annealing implementation.
    - `hsa.py`: Harmony Search Algorithm implementation.
    - `tsp_utils.py`: Utilities for distance calculation, instance generation, and plotting.
    - `experiment_runner.py`: Orchestrates large-scale experiments (80 trials).
    - `generate_report_plots.py`: Generates high-quality visualizations for analysis.
- `app/`: Interactive dashboard.
    - `main.py`: Streamlit application code.
- `experiment_results.json`: Detailed results from the experimental runs.
- `requirements.txt`: Python dependencies.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AzzamAlsharafi/TSP-Metaheuristic-Optimization.git
   cd TSP-Metaheuristic-Optimization
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Experiments
To execute the full suite of 80 trials (16 configurations × 5 trials) and save the results to `experiment_results.json`:
```bash
PYTHONPATH=. python3 src/experiment_runner.py
```

### 2. Launch the Streamlit App
To explore the algorithms interactively and view the experimental analysis:
```bash
streamlit run app/main.py
```

### 3. Generate Plots
To regenerate the static plots used for analysis:
```bash
PYTHONPATH=. python3 src/generate_report_plots.py
```

## Key Features
- **Interactive Visualization**: Real-time tour plotting and convergence trajectories in the Streamlit app.
- **Comprehensive Benchmarking**: Detailed analysis of solution quality, computational cost, and reproducibility.
- **Parameter Sensitivity**: Tools to analyze how different parameters (cooling rate, HMS, HMCR, etc.) affect performance.

## License
This project is for educational purposes as part of the TC6544 - Artificial Intelligence course.
