import json
import networkx as nx
import os

def select_file_from_folder(folder_path="graphs"):
    """
    Displays all files in the specified folder and allows the user to select one.

    Args:
        folder_path (str): Path to the folder containing files.

    Returns:
        str: Full path of the selected file.
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        print("The folder is empty!")
        return None

    # Show available files
    print("Available files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")

    # Ask the user to select a file
    while True:
        try:
            choice = int(input("Select the number of the file you want to open: "))
            if 1 <= choice <= len(files):
                filename = os.path.join(folder_path, files[choice-1])
                print(f"You selected: {filename}")
                return filename
            else:
                print(f"Enter a number between 1 and {len(files)}")
        except ValueError:
            print("Enter a valid number.")

def save_graph_json(G, filename):
    """
    Saves graph G in node-link JSON format, default key 'links'.
    Works only with isolated subgraphs.
    """
    # Make a copy for safety, no external references
    H = G.copy()
    data = nx.node_link_data(H, edges="edges")  # do not change edges="edges"

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_graph_json(filename, edges_key="edges"):
    """
    Loads a graph saved in node-link JSON format.

    Args:
        filename (str): Path to the JSON file
        edges_key (str): Key used for edges in JSON
                         default: "edges" (compatible with future NetworkX)
                         use "links" for older saves
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G_loaded = nx.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges=edges_key
    )

    return G_loaded

def check_save(target_company, subgraph):
    saved_graph = load_graph_json(f"graphs/{target_company}.json")
    # The graph you intended to save
    expected_graph = subgraph.copy()  # what you built in memory

    # --- Check nodes ---
    saved_nodes = set(saved_graph.nodes())
    expected_nodes = set(expected_graph.nodes())

    missing_nodes = expected_nodes - saved_nodes
    extra_nodes = saved_nodes - expected_nodes

    print("Missing nodes:", missing_nodes)
    print("Extra nodes:", extra_nodes)

    # --- Check edges ---
    saved_edges = set((u, v) for u, v in saved_graph.edges())
    expected_edges = set((u, v) for u, v in expected_graph.edges())

    missing_edges = expected_edges - saved_edges
    extra_edges = saved_edges - expected_edges

    print("Missing edges:", missing_edges)
    print("Extra edges:", extra_edges)

    # --- Total verification ---
    if not missing_nodes and not extra_nodes and not missing_edges and not extra_edges:
        print("The saved graph perfectly matches the expected graph!")
    else:
        print("There are differences between the saved graph and the expected graph.")

def calculate_result_and_owners(graph):
    incoming_only_nodes = [n for n in graph.nodes if graph.in_degree(n) > 0 and graph.out_degree(n) == 0]
    result = []
    owners = []

    for n in incoming_only_nodes:
        incoming_edges = graph.in_edges(n, data=True)
        total_weights = sum(float(d.get("weight", 0) or 0) for _, _, d in incoming_edges)
        node_name = graph.nodes[n].get("name", n)

        total_percentage = government_percentage = 0.0
        for u, _, d in incoming_edges:
            weight = float(d.get("weight", 0) or 0)
            source_name = graph.nodes[u].get("name", str(u))
            owners.append({"company_name": source_name, "permid": u, "percentage": weight})
            total_percentage += weight
            if "government" in source_name.lower():
                government_percentage += weight

        result.append({
            "node_id": n,
            "name": node_name,
            "total_percentage": round(total_weights, 2),
            "float": 100 - round(total_weights, 2),
            "government_percentage": round(government_percentage, 2),
        })
    return result, owners

def create_subgraph(G, target):
    # Get all predecessors
    predecessors = nx.ancestors(G, target)
    predecessors.add(target)  # Include the target node

    # Create a subgraph
    subgraph = G.subgraph(predecessors).copy()
    return subgraph

def analyze_graph(G: nx.DiGraph, target=None):
    """
    Analyzes a directed graph and calculates general and target-specific metrics.

    Args:
        G (nx.DiGraph): NetworkX directed graph
        target (any): Target node to analyze (must exist in G)

    Returns:
        dict: Dictionary with calculated metrics
    """

    metrics = {}

    # --- GENERAL METRICS ---
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['average_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Total edge weights
    total_weight = sum([data.get('weight', 1) for _, _, data in G.edges(data=True)])
    metrics['total_weight'] = total_weight

    # --- TARGET-SPECIFIC METRICS ---
    if target is not None and target in G.nodes:
        in_edges = G.in_edges(target, data=True)
        out_edges = G.out_edges(target, data=True)

        weight_in = sum([d.get('weight', 1) for _, _, d in in_edges])
        weight_out = sum([d.get('weight', 1) for _, _, d in out_edges])

        metrics['incoming_weight'] = weight_in
        metrics['outgoing_weight'] = weight_out
        metrics['incoming_weight_percentage'] = (weight_in / total_weight * 100) if total_weight > 0 else 0
        metrics['outgoing_weight_percentage'] = (weight_out / total_weight * 100) if total_weight > 0 else 0

        metrics['in_degree'] = G.in_degree(target)
        metrics['out_degree'] = G.out_degree(target)

        metrics['connected_nodes_in'] = len(set([u for u, _, _ in in_edges]))
        metrics['connected_nodes_out'] = len(set([v for _, v, _ in out_edges]))
    else:
        metrics['note'] = "No target specified or target not present in the graph."

    return metrics

def filter_graph(G, target, threshold_target=0.20, threshold_others=5):
    subgraph = G.copy()

    # Iterate through all edges with weights
    edges_to_remove = []
    for u, v, d in subgraph.edges(data=True):
        weight = d.get('weight', 0)

        # Case 1️⃣: outgoing edge from target → always remove
        if u == target:
            edges_to_remove.append((u, v))

        # Case 2️⃣: incoming edge to target → keep only if weight > threshold_target
        elif v == target:
            if weight <= threshold_target:
                edges_to_remove.append((u, v))

        # Case 3️⃣: edge not connected to target → keep only if weight > threshold_others
        else:
            if weight <= threshold_others:
                edges_to_remove.append((u, v))

    # Remove edges that do not meet criteria
    subgraph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes (no edges)
    isolated_nodes = list(nx.isolates(subgraph))
    subgraph.remove_nodes_from(isolated_nodes)

    # Create the subgraph relative to the target
    subgraph = create_subgraph(subgraph, target)
    return subgraph
