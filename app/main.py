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
)
from src.sa import SimulatedAnnealing
from src.hsa import HarmonySearch

st.set_page_config(page_title="TSP Metaheuristic Optimization", layout="wide")

st.title("TSP Metaheuristic Optimization")
st.markdown("""
This app explores two metaheuristic algorithms for solving the Traveling Salesman Problem (TSP):
**Simulated Annealing (SA)** and **Harmony Search Algorithm (HSA)**.
""")


# Load or generate instance
@st.cache_data
def get_instance():
    cities = generate_tsp_instance(100)
    dist_matrix = calculate_distance_matrix(cities)
    return cities, dist_matrix


cities, dist_matrix = get_instance()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")
    algo_choice = st.selectbox(
        "Select Algorithm", ["Simulated Annealing", "Harmony Search"]
    )

    if algo_choice == "Simulated Annealing":
        temp = st.slider("Initial Temperature", 100, 10000, 1000)
        cooling = st.slider("Cooling Rate", 0.9, 0.9999, 0.995, format="%.4f")
        iters = st.number_input("Max Iterations", 1000, 100000, 10000)

        if st.button("Run SA"):
            sa = SimulatedAnnealing(
                dist_matrix, initial_temp=temp, cooling_rate=cooling, max_iter=iters
            )
            with st.spinner("Running SA..."):
                best_path, best_dist, history, exec_time = sa.solve()
            st.session_state["result"] = (
                best_path,
                best_dist,
                history,
                exec_time,
                "SA",
            )

    else:
        hms = st.slider("Harmony Memory Size (HMS)", 5, 50, 20)
        hmcr = st.slider("HM Considering Rate (HMCR)", 0.5, 0.99, 0.9, format="%.2f")
        par = st.slider("Pitch Adjusting Rate (PAR)", 0.1, 0.9, 0.3, format="%.2f")
        iters = st.number_input("Max Iterations", 100, 20000, 5000)

        if st.button("Run HSA"):
            hsa = HarmonySearch(
                dist_matrix, hms=hms, hmcr=hmcr, par=par, max_iter=iters
            )
            with st.spinner("Running HSA..."):
                best_path, best_dist, history, exec_time = hsa.solve()
            st.session_state["result"] = (
                best_path,
                best_dist,
                history,
                exec_time,
                "HSA",
            )

with col2:
    st.header("Visualization")
    if "result" in st.session_state:
        best_path, best_dist, history, exec_time, name = st.session_state["result"]

        st.subheader(f"Best Solution ({name})")
        st.write(f"**Total Distance:** {best_dist:.2f}")
        st.write(f"**Execution Time:** {exec_time:.2f}s")

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
            x=cities[:, 0], y=cities[:, 1], title="TSP Instance (100 Cities)"
        )
        fig_inst.update_layout(height=500)
        st.plotly_chart(fig_inst, use_container_width=True)

st.divider()
st.header("Detailed Experimental Analysis")

if os.path.exists("experiment_results.json"):
    with open("experiment_results.json", "r") as f:
        results = json.load(f)

    st.subheader("1. Performance & Stability Analysis (80 Trials)")
    st.markdown("""
    This section analyzes the performance and stability of both algorithms across **16 different parameter configurations** (8 for SA, 8 for HSA). 
    Each configuration was run for **5 independent trials** to assess reproducibility.
    """)

    all_data = []
    for algo_type in ["sa_configs", "hsa_configs"]:
        algo_name = "SA" if algo_type == "sa_configs" else "HSA"
        for config in results[algo_type]:
            params_str = ", ".join([f"{k}={v}" for k, v in config["params"].items()])
            for trial in config["trials"]:
                all_data.append(
                    {
                        "Algorithm": algo_name,
                        "Config ID": config["config_id"],
                        "Parameters": params_str,
                        "Best Distance": trial["best_dist"],
                        "Execution Time (s)": trial["exec_time"],
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
        "Below is the raw data for all 80 trials, including the specific parameters used for each run."
    )

    st.dataframe(
        df[
            [
                "Algorithm",
                "Config ID",
                "Parameters",
                "Best Distance",
                "Execution Time (s)",
            ]
        ],
        use_container_width=True,
    )

else:
    st.warning(
        "Experiment results not found. Run `src/experiment_runner.py` to generate them."
    )
