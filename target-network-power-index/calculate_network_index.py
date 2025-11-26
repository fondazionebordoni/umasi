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

# Calculate T-NPI and T-NPF for different scenarios
scenarios = ['1', '2', '3', '4']
results = {}
for scenario in scenarios:
    subgraph = scenario_chosen_all(scenario, graph, target)

    # 🔹 Calculate indexes
    NPI_global, NPF_global, NPI_to_E, NPF_to_E, labels = calculate_index(subgraph, target)
    weight_df = calculate_shareholdings(subgraph, target, verbose=False)

    # 🔹 Save results into DataFrames
    npi_df = save_index(NPI_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPI')
    npf_df = save_index(NPF_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPF')
    npi_df['NPI'] = npi_df['NPI'] * 100
    npf_df['NPF'] = npf_df['NPF'] * 100

    weight_df['Name'] = weight_df['PermID'].apply(lambda x: subgraph.nodes[x].get('name', ''))
    weight_df = weight_df.sort_values(by='Percentage', ascending=False)

    ultimate_owner = npi_df['Name'][0]
    ultimate_owner_percent = npi_df['Shareholding'][0]
    npi = npi_df['NPI'][0]

    subgraph = add_index_to_graph(npi_df, npf_df, subgraph)

    # Save subgraph as JSON
    save_graph_json(subgraph, f'output/{controlled}_{scenario}.json')

    # Save interactive graphs in HTML (PyVis)
    graphname = save_npf_graph_html_pyvis(
        subgraph, npf_df, target, npi_df['ID'][0], npi_df['NPI'][0] * 100,
        f"html/graph_{controlled}_{scenario}.html"
    )
    graphname_anonymized = save_npf_graph_html_pyvis_anonymized(
        subgraph, npf_df, target, npi_df['ID'][0], npi_df['NPI'][0] * 100,
        f"html/graph_{controlled}_{scenario}_anonymized.html"
    )

    # 🔹 Collect results
    result = {
        'subgraph': subgraph,
        'NPI_global': NPI_global,
        'NPF_global': NPF_global,
        'weight_df': weight_df,
        'NPI_df': npi_df,
        'NPF_df': npf_df,
        'labels': labels
    }

    results[scenario] = result

# Create the plots
df_top_shareholder = top_shareholders(results)
df_plot_npi, df_plot_npf = plot_mizuno_index(
    results,
    target_name=controlled,
    font_size=35,
    linewidth=5,
    markersize=12
)