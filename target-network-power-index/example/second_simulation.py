import os
import sys
import pandas as pd

# Add project root (current folder) to path to import custom modules
project_root = os.path.abspath(os.path.join("."))
sys.path.append(project_root)

from network_power_indexes.calculate_index import *
from network_power_indexes.utils import *
from network_power_indexes.plots import *
from network_power_indexes.simulations import *

filename = "graphs/example.json"

# Load the selected file into a networkx graph
G = load_graph_json(filename)
graph = G.copy()

# Check which company is the target
result, owners = calculate_result_and_owners(G)
try:
    target = result[0]['node_id']
    controlled = result[0]['name']
    print(f'Found company {controlled}')
except:
    print('Error')

# Check direct shareholders
direct_shareholders = [
    (u, G.nodes[u].get('name', None), G[u][target].get('weight', None))
    for u in G.predecessors(target)
]
df_shareholders = pd.DataFrame(direct_shareholders, columns=['ID', 'Name', 'Percentage'])
df_shareholders = df_shareholders.sort_values(by='Percentage', ascending=False)

# Run simulation on the conglomerate of major US funds
scenario = '2'
indices = find_interconnected_groups(G, target)

companies_analyzed = df_shareholders[df_shareholders['ID'].isin(indices)]['Name'].tolist()

# 🔹 Initial shares for each of the three companies
start_shares = {
    company: df_shareholders.loc[df_shareholders['Name'] == company]['Percentage'].iloc[0]
    for company in companies_analyzed
}

share_increase = 0.5
mean_increase = 0
results = {}
ultimate_owner = "Company 1"

# 🔹 Loop until ALL companies collectively surpass the threshold
while ultimate_owner == "Company 1" and all(share < 25 for share in start_shares.values()):

    # 🔹 Apply simultaneous share changes
    graph_simulated, df_shareholders_simulated = change_multishares(
        graph,
        df_shareholders,
        target,
        companies_analyzed,
        start_shares
    )

    subgraph = scenario_chosen_all(scenario, graph_simulated, target)

    # 🔹 Calculate indexes
    NPI_global, NPF_global, NPI_to_E, NPF_to_E, labels = calculate_index(subgraph, target)
    weight_df = calculate_shareholdings(subgraph, target, verbose=False)

    # 🔹 Save results into DataFrames
    npi_df = save_index(NPI_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPI')
    npf_df = save_index(NPF_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPF')

    npi_df["NPI"] *= 100
    npf_df["NPF"] *= 100

    weight_df["Names"] = weight_df["PermID"].apply(lambda x: subgraph.nodes[x].get("name", ""))
    weight_df = weight_df.sort_values(by="Percentage", ascending=False)

    # 🔹 Ultimate owner
    ultimate_owner = npi_df["Name"].iloc[0]
    ultimate_owner_percent = npi_df["Shareholding"].iloc[0]
    npi = npi_df["NPI"].iloc[0]

    subgraph = add_index_to_graph(npi_df, npf_df, subgraph)

    name_suffix = "_".join([f"{c}_{start_shares[c]}" for c in companies_analyzed])

    # 🔹 Save results
    results[str(mean_increase)] = {
        "subgraph": subgraph,
        "NPI_global": NPI_global,
        "NPF_global": NPF_global,
        "weight_df": weight_df,
        "NPI_df": npi_df,
        "NPF_df": npf_df,
        "labels": labels
    }
    mean_increase += share_increase

    # 🔹 Increase shares simultaneously
    for company in start_shares:
        start_shares[company] += share_increase


# Create the plot
uo1_name = "Company 1"
uo2_name = "Company 2"

plot_share_and_npi(results, companies_analyzed, uo1_name, uo2_name, font_size=16)