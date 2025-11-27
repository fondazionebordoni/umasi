import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root (current folder) to path to import custom modules
project_root = os.path.abspath(os.path.join("."))
sys.path.append(project_root)

from network_power_indexes.calculate_index import *
from network_power_indexes.utils import *
from network_power_indexes.plots import *
from network_power_indexes.simulations import *

# Select a file from the folder
filename = select_file_from_folder()

# Load the selected file into a NetworkX graph
G = load_graph_json(filename)
graph = G.copy()

# Check which company is the target
result, owners = calculate_result_and_owners(G)
try:
    target = result[0]['node_id']
    controlled = result[0]['name']
    print(f'Found company {controlled}')
except:
    print('Error identifying target company')

# Check direct shareholders
direct_shareholders = [
    (u, G.nodes[u].get('name', None), G[u][target].get('weight', None))
    for u in G.predecessors(target)
]
df_shareholders = pd.DataFrame(direct_shareholders, columns=['ID', 'Name', 'Percentage'])
df_shareholders = df_shareholders.sort_values(by='Percentage', ascending=False)

# Initialize a list to store all NPI/NPF values in long format
rows = []

# Dictionaries to save NPI and NPF for each T
npi_series = {}
npf_series = {}

# Define the range of Monte Carlo iterations T
T_values = range(1000, 20001, 1000)

# Loop over all T values
for T in T_values:
    print(f"▶ Processing T = {T}")

    # 🔹 Generate the subgraph for the scenario (here scenario '4')
    subgraph = scenario_chosen_all('4', graph, target)

    # 🔹 Compute NPI and NPF indices for the given T
    NPI_global, NPF_global, NPI_to_E, NPF_to_E, labels = calculate_index(subgraph, target, T=T)

    # 🔹 Compute the shareholding weights for nodes
    weight_df = calculate_shareholdings(subgraph, target, verbose=False)

    # 🔹 Save the NPI and NPF indices as DataFrames
    npi_df = save_index(NPI_to_E, labels, subgraph, weight_df, target, controlled, T, index='NPI')
    npf_df = save_index(NPF_to_E, labels, subgraph, weight_df, target, controlled, T, index='NPF')

    # 🔹 Convert indices to percentages
    npi_df['NPI'] = npi_df['NPI'] * 100
    npf_df['NPF'] = npf_df['NPF'] * 100

    # 🔹 Store each node's NPI and NPF in the long-format list
    for idx, row in npi_df.iterrows():
        node = row["ID"]
        NPI = row["NPI"]
        NPF = npf_df.loc[idx, "NPF"]

        rows.append({
            "T": T,
            "node": node,
            "NPI": NPI,
            "NPF": NPF
        })

    # 🔹 Save NPI and NPF in dictionaries keyed by T for easy access
    npi_series[T] = dict(zip(npi_df["ID"], npi_df["NPI"]))
    npf_series[T] = dict(zip(npf_df["ID"], npf_df["NPF"]))

# 🔹 Convert the list of all observations into a pandas DataFrame
df_long = pd.DataFrame(rows)


def validation(series, index='NPI'):
    """
    Evaluate the convergence and stability of NPI or NPF series across Monte Carlo iterations.

    Parameters
    ----------
    series : dict
        Dictionary keyed by T, where each value is a dict {node_id: NPI/NPF}.
    index : str, optional
        Name of the index being validated ('NPI' or 'NPF'), by default 'NPI'.

    Returns
    -------
    RMSE_global : float
        Root Mean Square Error across all nodes and T < 20000 vs benchmark T=20000.
    mean_diff : float
        Mean difference from benchmark.
    std_diff : float
        Standard deviation of differences.
    diffs : np.ndarray
        Array of all deviations used to compute RMSE.
    """

    # 🔹 Benchmark values (T = 20,000)
    np_Tmax = series[20000]

    diffs = []  # Will store all deviations

    # 🔹 Compare each T < 20,000 against the benchmark
    for T, np_dict in series.items():
        if T == 20000:
            continue  # Skip the benchmark itself
        for node_id, np_value in np_dict.items():
            np_final = np_Tmax[node_id]
            diffs.append(np_value - np_final)

    diffs = np.array(diffs)

    # 🔹 Compute global RMSE, mean, and standard deviation of deviations
    RMSE_global = np.sqrt(np.mean(diffs ** 2))
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # 🔹 Print summary
    print(f"📊 {index} VARIATION AS T CHANGES")
    print(f"Global RMSE      : {RMSE_global:.6f}")
    print(f"Mean difference  : {mean_diff:.6f}")
    print(f"Std deviation    : {std_diff:.6f}")

    return RMSE_global, mean_diff, std_diff, diffs

# Validate NPI stability
RMSE_NPI, mean_NPI, std_NPI, diffs_NPI = validation(npi_series, index='NPI')

# Validate NPF stability
RMSE_NPF, mean_NPF, std_NPF, diffs_NPF = validation(npf_series, index='NPF')

# 🔹 Identify the top 5 nodes based on average NPI across all T
top5_nodes = (
    df_long.groupby("node")["NPI"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

# 🔹 Filter the long-format DataFrame to include only the top 5 nodes
df_top5 = df_long[df_long["node"].isin(top5_nodes)]

# =======================
# --- PLOT NPI ---
# =======================
plt.figure(figsize=(24, 12))

for node in top5_nodes:
    subset = df_top5[df_top5["node"] == node]
    plt.plot(subset["T"], subset["NPI"], marker='o', label=f"Node {node}")

plt.title("NPI of the Top 5 Nodes as T Varies", fontsize=28)
plt.xlabel("T (Monte Carlo Iterations)", fontsize=26)
plt.ylabel("NPI", fontsize=26)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.show()

# =======================
# --- PLOT NPF ---
# =======================
plt.figure(figsize=(24, 12))

for node in top5_nodes:
    subset = df_top5[df_top5["node"] == node]
    plt.plot(subset["T"], subset["NPF"], marker='o', label=f"Node {node}")

plt.title("NPF of the Top 5 Nodes as T Varies", fontsize=28)
plt.xlabel("T (Monte Carlo Iterations)", fontsize=26)
plt.ylabel("NPF", fontsize=26)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.show()