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

filename = select_file_from_folder()

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

# Consider scenario 2 for running single simulations on the top 4 shareholders of the target company
scenario = '2'
top_shareholders = df_shareholders.head(5).reset_index(drop=True)

# Remove the row if the name is "Company 1"
top_shareholders = top_shareholders[top_shareholders['Name'] != "Company 1"].reset_index(drop=True)

all_results = {}
for company_analyzed in top_shareholders['Name'].to_list():
    share = top_shareholders.loc[top_shareholders['Name'] == company_analyzed]['Percentage']
    start_share = share[share.index[0]]
    share_increase = 2
    results = {}
    ultimate_owner = 'Company 1'
    while ultimate_owner == 'Company 1' and start_share < 25:
        graph_simulated, df_shareholders_simulated = change_shares(graph, df_shareholders, target, company_analyzed, start_share)
        subgraph = scenario_chosen_all(scenario, graph_simulated, target)

        # 🔹 Calculate indexes
        NPI_global, NPF_global, NPI_to_E, NPF_to_E, labels = calculate_index(subgraph, target)
        weight_df = calculate_shareholdings(subgraph, target, verbose=False)

        # 🔹 Save results in DataFrames
        npi_df = save_index(NPI_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPI')
        npf_df = save_index(NPF_to_E, labels, subgraph, weight_df, target, controlled, scenario, index='NPF')
        npi_df['NPI'] = npi_df['NPI'] * 100
        npf_df['NPF'] = npf_df['NPF'] * 100

        weight_df['Names'] = weight_df['PermID'].apply(lambda x: subgraph.nodes[x].get('name', ''))
        weight_df = weight_df.sort_values(by='Percentage', ascending=False)

        ultimate_owner = npi_df['Name'][0]
        ultimate_owner_percent = npi_df['Shareholding'][0]
        npi = npi_df['NPI'][0]

        subgraph = add_index_to_graph(npi_df, npf_df, subgraph)

        save_graph_json(subgraph, f'simulations/first/{controlled}_{scenario}_simulated_{start_share}_{company_analyzed}.json')

        graphname = save_npf_graph_html_pyvis(
            subgraph, npf_df, target, npi_df['ID'][0], npi_df['NPI'][0] * 100,
            f"html/graph_{controlled}_{scenario}_simulated_{start_share}_{company_analyzed}.html"
        )
        graphname_anonymized = save_npf_graph_html_pyvis_anonymized(
            subgraph, npf_df, target, npi_df['ID'][0], npi_df['NPI'][0] * 100,
            f"html/graph_{controlled}_{scenario}_simulated_{start_share}_{company_analyzed}_anonymized.html"
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

        results[str(start_share)] = result
        start_share += share_increase
    all_results[company_analyzed] = results

# Create the plots
plot_ultimate_owner_evolution(all_results, controlled, font_size=26)