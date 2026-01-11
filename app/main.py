import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

# Add the root directory to sys.path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tsp_utils import (
    generate_tsp_instance,
    calculate_distance_matrix,
    get_optimal_distance,
)
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch

st.set_page_config(page_title="TSP Metaheuristic Optimization", layout="wide")

st.title("TSP Metaheuristic Optimization")
st.markdown("""
This app explores two metaheuristic algorithms for solving the Traveling Salesman Problem (TSP):
**Simulated Annealing (SA)** and **Harmony Search Algorithm (HSA)**.

**Benchmark:** rd100 (100-city random TSP by Reinelt) | **Known Optimal Distance:** 7910
""")


# Load or generate instance
# Uses rd100 TSPLIB benchmark for consistency
@st.cache_data
def get_instance():
    cities = generate_tsp_instance()  # Loads from data/rd100.tsp
    dist_matrix = calculate_distance_matrix(cities)
    optimal_distance = get_optimal_distance()
    return cities, dist_matrix, optimal_distance


cities, dist_matrix, optimal_distance = get_instance()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")
    algo_choice = st.selectbox(
        "Select Algorithm", ["Simulated Annealing", "Harmony Search"]
    )

    # Random seed input (common for both algorithms)
    random_seed = st.number_input(
        "Random Seed (for reproducibility)",
        min_value=0,
        max_value=10000,
        value=42,
        help="Use the same seed to get reproducible results"
    )

    if algo_choice == "Simulated Annealing":
        temp = st.number_input("Initial Temperature", min_value=0.0000001, max_value=10000.0, value=1000.0, step=0.0000001, format="%.7f")
        cooling = st.number_input("Cooling Rate", min_value=0.9, max_value=0.999999999, value=0.995, step=0.000000001, format="%.9f")
        min_temp = st.number_input("Min Temperature (0 = no early stop)", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f",
                                    help="Set to 0 to disable early stopping and run full max_iter")
        iters = st.number_input("Max Iterations", min_value=1000, max_value=10000000, value=50000, step=1000)

        if st.button("Run SA"):
            sa = SimulatedAnnealing(
                dist_matrix, initial_temp=temp, cooling_rate=cooling, min_temp=min_temp, max_iter=int(iters), random_seed=int(random_seed)
            )
            with st.spinner("Running SA..."):
                best_path, best_dist, history, exec_time, metadata = sa.solve()
            st.session_state["result"] = (
                best_path,
                best_dist,
                history,
                exec_time,
                "SA",
                metadata,
            )

    else:
        hms = st.number_input("Harmony Memory Size (HMS)", min_value=5, max_value=50, value=20, step=1)
        hmcr = st.number_input("HM Considering Rate (HMCR)", min_value=0.5, max_value=0.99, value=0.9, step=0.01, format="%.2f")
        par = st.number_input("Pitch Adjusting Rate (PAR)", min_value=0.1, max_value=0.9, value=0.3, step=0.01, format="%.2f")
        iters = st.number_input("Max Iterations", min_value=1000, max_value=10000000, value=10000, step=1000)

        if st.button("Run HSA"):
            hsa = HarmonySearch(
                dist_matrix, hms=hms, hmcr=hmcr, par=par, max_iter=int(iters), random_seed=int(random_seed)
            )
            with st.spinner("Running HSA..."):
                best_path, best_dist, history, exec_time = hsa.solve()
            st.session_state["result"] = (
                best_path,
                best_dist,
                history,
                exec_time,
                "HSA",
                None,  # HSA doesn't have metadata yet
            )

with col2:
    st.header("Visualization")
    if "result" in st.session_state:
        best_path, best_dist, history, exec_time, name, metadata = st.session_state["result"]

        st.subheader(f"Best Solution ({name})")
        st.write(f"**Total Distance:** {best_dist:.2f}")
        gap = ((best_dist - optimal_distance) / optimal_distance) * 100
        st.write(f"**Gap from Optimal (7910):** {gap:.2f}%")
        st.write(f"**Execution Time:** {exec_time:.2f}s")

        # Display early stopping info for SA
        if metadata is not None:
            st.write(f"**Iterations Completed:** {metadata['iterations_completed']:,}")
            if metadata['early_stopped']:
                st.warning(f"⚠️ **Early Stopping:** Algorithm stopped at iteration {metadata['iterations_completed']:,} (temp reached {metadata['final_temperature']:.6f})")
            else:
                st.success(f"✓ **No Early Stopping:** Completed full {metadata['iterations_completed']:,} iterations")
            st.write(f"**Final Temperature:** {metadata['final_temperature']:.6f}")

        # Plot Solution
        fig_sol = go.Figure()
        fig_sol.add_trace(
            go.Scatter(
                x=cities[:, 0],
                y=cities[:, 1],
                mode="markers",
                name="Cities",
                marker=dict(color="red", size=8),
            )
        )

        path_x = [cities[i, 0] for i in best_path] + [cities[best_path[0], 0]]
        path_y = [cities[i, 1] for i in best_path] + [cities[best_path[0], 1]]
        fig_sol.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines",
                name="Tour",
                line=dict(color="blue", width=2),
            )
        )

        fig_sol.update_layout(
            title="Optimal Tour Visualization",
            xaxis_title="X",
            yaxis_title="Y",
            height=500,
        )
        st.plotly_chart(fig_sol, use_container_width=True)

        # Plot Convergence
        st.subheader("Convergence Trajectory")
        fig_conv = px.line(
            x=range(len(history)),
            y=history,
            labels={"x": "Iteration", "y": "Best Distance"},
        )
        fig_conv.update_layout(height=400)
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Run an algorithm to see the results.")
        # Show instance
        fig_inst = px.scatter(
            x=cities[:, 0], y=cities[:, 1], title="rd100 TSP Instance (100 Cities, Optimal: 7910)"
        )
        fig_inst.update_layout(height=500)
        st.plotly_chart(fig_inst, use_container_width=True)

st.divider()
st.header("Detailed Experimental Analysis")

if os.path.exists("experiment_results.json"):
    with open("experiment_results.json", "r") as f:
        results = json.load(f)

    st.subheader("1. Performance & Stability Analysis (90 Trials)")
    st.markdown("""
    This section analyzes the performance and stability of both algorithms across **18 different parameter configurations** (9 for SA, 9 for HSA).
    Each configuration was run for **5 independent trials** to assess reproducibility.
    """)

    all_data = []
    for algo_type in ["sa_configs", "hsa_configs"]:
        algo_name = "SA" if algo_type == "sa_configs" else "HSA"
        for config in results[algo_type]:
            params_str = ", ".join([f"{k}={v}" for k, v in config["params"].items()])
            for trial in config["trials"]:
                # SA trials use "seed", HSA trials use "random_seed"
                seed_value = trial.get("seed") or trial.get("random_seed", "N/A")

                all_data.append(
                    {
                        "Algorithm": algo_name,
                        "Config ID": config["config_id"],
                        "Parameters": params_str,
                        "Best Distance": trial["best_dist"],
                        "Execution Time (s)": trial["exec_time"],
                        "Random Seed": seed_value,
                    }
                )
    df = pd.DataFrame(all_data)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Solution Quality (Distance)**")
        # Split Distance Boxplots
        fig_dist_sa = px.box(
            df[df["Algorithm"] == "SA"],
            x="Config ID",
            y="Best Distance",
            title="SA: Distance Distribution",
            color_discrete_sequence=["blue"],
        )
        st.plotly_chart(fig_dist_sa, use_container_width=True)

        fig_dist_hsa = px.box(
            df[df["Algorithm"] == "HSA"],
            x="Config ID",
            y="Best Distance",
            title="HSA: Distance Distribution",
            color_discrete_sequence=["green"],
        )
        st.plotly_chart(fig_dist_hsa, use_container_width=True)

        # Summary Stats
        stats = (
            df.groupby(["Algorithm", "Config ID"])["Best Distance"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        st.write("**Stability Metrics (Distance)**")
        st.dataframe(stats, use_container_width=True)

    with c2:
        st.write("**Computational Cost (Efficiency)**")
        # Split Time Boxplots
        fig_time_sa = px.box(
            df[df["Algorithm"] == "SA"],
            x="Config ID",
            y="Execution Time (s)",
            title="SA: Execution Time",
            color_discrete_sequence=["blue"],
        )
        st.plotly_chart(fig_time_sa, use_container_width=True)

        fig_time_hsa = px.box(
            df[df["Algorithm"] == "HSA"],
            x="Config ID",
            y="Execution Time (s)",
            title="HSA: Execution Time",
            color_discrete_sequence=["green"],
        )
        st.plotly_chart(fig_time_hsa, use_container_width=True)

        # Time Stats
        time_stats = (
            df.groupby(["Algorithm", "Config ID"])["Execution Time (s)"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        st.write("**Efficiency Metrics (Time)**")
        st.dataframe(time_stats, use_container_width=True)

    st.subheader("2. Reproducibility & Raw Trial Data")
    st.markdown(
        "Below is the raw data for all 90 trials, including the specific parameters used for each run."
    )

    st.dataframe(
        df[
            [
                "Algorithm",
                "Config ID",
                "Parameters",
                "Best Distance",
                "Execution Time (s)",
                "Random Seed",
            ]
        ],
        use_container_width=True,
    )

else:
    st.warning(
        "Experiment results not found. Run `src/experiment_runner.py` to generate them."
    )

st.divider()
st.header("Phase 2: Best Configuration Comparison")

if os.path.exists("best_config_results.json"):
    with open("best_config_results.json", "r") as f:
        phase2_results = json.load(f)

    st.markdown("""
    This section shows the **statistical validation** of the best SA and HSA configurations.
    Each algorithm's best configuration was run **30 times** for robust statistical comparison.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Best SA: {phase2_results['sa']['config_id']}")
        st.write(f"**Parameters:** {phase2_results['sa']['params']}")
        stats_sa = phase2_results['sa']['statistics']
        st.metric("Mean Distance", f"{stats_sa['mean']:.2f}")
        st.metric("Std Deviation", f"{stats_sa['std']:.2f}")
        st.write(f"**Min:** {stats_sa['min']:.2f}")
        st.write(f"**Max:** {stats_sa['max']:.2f}")
        st.write(f"**95% CI:** [{stats_sa['ci_95'][0]:.2f}, {stats_sa['ci_95'][1]:.2f}]")
        st.write(f"**Avg Time:** {stats_sa['mean_time']:.2f}s")

    with col2:
        st.subheader(f"Best HSA: {phase2_results['hsa']['config_id']}")
        st.write(f"**Parameters:** {phase2_results['hsa']['params']}")
        stats_hsa = phase2_results['hsa']['statistics']
        st.metric("Mean Distance", f"{stats_hsa['mean']:.2f}")
        st.metric("Std Deviation", f"{stats_hsa['std']:.2f}")
        st.write(f"**Min:** {stats_hsa['min']:.2f}")
        st.write(f"**Max:** {stats_hsa['max']:.2f}")
        st.write(f"**95% CI:** [{stats_hsa['ci_95'][0]:.2f}, {stats_hsa['ci_95'][1]:.2f}]")
        st.write(f"**Avg Time:** {stats_hsa['mean_time']:.2f}s")

    st.divider()
    st.subheader("Statistical Significance Test")

    tests = phase2_results['statistical_tests']

    col1, col2, col3 = st.columns(3)
    col1.metric("t-statistic", f"{tests['t_statistic']:.4f}")
    # Use scientific notation for very small p-values
    p_val_str = f"{tests['p_value']:.2e}" if tests['p_value'] < 0.0001 else f"{tests['p_value']:.4f}"
    col2.metric("p-value", p_val_str)
    col3.metric("Cohen's d", f"{tests['cohens_d']:.2f}")

    if tests['significant']:
        st.success(f"✓ **{tests['better_algorithm']} is SIGNIFICANTLY better** (p < 0.05)")
        worse_algo = 'HSA' if tests['better_algorithm'] == 'SA' else 'SA'
        st.write(f"{worse_algo} solutions were **{tests['percentage_difference']:.2f}% worse** than {tests['better_algorithm']} on average")
    else:
        st.info("No significant difference between algorithms (p >= 0.05)")

    # Boxplot comparison - Split into two plots for better visibility
    st.subheader("Distribution Comparison")

    sa_dists = [trial['best_dist'] for trial in phase2_results['sa']['trials']]
    hsa_dists = [trial['best_dist'] for trial in phase2_results['hsa']['trials']]

    import plotly.graph_objects as go

    col1, col2 = st.columns(2)

    with col1:
        # SA Boxplot with its own scale
        fig_sa = go.Figure()
        fig_sa.add_trace(go.Box(
            y=sa_dists,
            name=phase2_results['sa']['config_id'],
            marker_color='blue',
            boxmean='sd',
            showlegend=False
        ))

        # Add optimal line
        fig_sa.add_hline(y=7910, line_dash="dash", line_color="red",
                        annotation_text="Optimal (7910)", annotation_position="top right")

        fig_sa.update_layout(
            title=f"{phase2_results['sa']['config_id']} Distribution",
            yaxis_title="Tour Distance",
            height=500,
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                range=[8500, 11000]  # Custom range to show optimal and SA distribution
            )
        )
        st.plotly_chart(fig_sa, use_container_width=True)

        # Show statistics
        st.write(f"**Mean:** {phase2_results['sa']['statistics']['mean']:.2f}")
        st.write(f"**Gap from Optimal:** {phase2_results['sa']['statistics']['mean'] - 7910:.2f} ({((phase2_results['sa']['statistics']['mean'] - 7910)/7910*100):.1f}%)")

    with col2:
        # HSA Boxplot with its own scale
        fig_hsa = go.Figure()
        fig_hsa.add_trace(go.Box(
            y=hsa_dists,
            name=phase2_results['hsa']['config_id'],
            marker_color='green',
            boxmean='sd',
            showlegend=False
        ))

        # Optimal line not shown for HSA (too far below the range)

        fig_hsa.update_layout(
            title=f"{phase2_results['hsa']['config_id']} Distribution",
            yaxis_title="Tour Distance",
            height=500,
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                range=[43000, 48000]  # Custom range to show HSA distribution clearly
            )
        )
        st.plotly_chart(fig_hsa, use_container_width=True)

        # Show statistics
        st.write(f"**Mean:** {phase2_results['hsa']['statistics']['mean']:.2f}")
        st.write(f"**Gap from Optimal:** {phase2_results['hsa']['statistics']['mean'] - 7910:.2f} ({((phase2_results['hsa']['statistics']['mean'] - 7910)/7910*100):.1f}%)")

else:
    st.info(
        "Phase 2 results not found. After running Phase 1, run `PYTHONPATH=. python3 src/best_config_comparison.py` to generate Phase 2 results."
    )
