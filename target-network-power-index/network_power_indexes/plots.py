# ---------------------------
# Libraries for graphs and visualization
# ---------------------------
import networkx as nx
from pyvis.network import Network
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_hex
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


def display_graph(subgraph):
    """Plot a network graph using NetworkX and Matplotlib."""

    fig, ax = plt.subplots(figsize=(40, 40))
    pos = nx.spring_layout(subgraph, seed=42, k=10)  # automatic layout

    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_color='skyblue', node_size=3000)

    # Draw labels (using 'name' attribute)
    labels = nx.get_node_attributes(subgraph, 'name')
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=18, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, width=2, arrowstyle='-|>', arrowsize=60, ax=ax)

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=18, ax=ax)

    plt.axis('off')
    plt.show()



# ---------------------------
# Functions for normal plots
# ---------------------------
def top_shareholders(results, anonymized=True):
    """Plot top 5 shareholders per scenario."""

    records = []

    for scenario, data in results.items():
        weights_df = data['weight_df'].copy()
        weights_df = weights_df.sort_values(by='Percentage', ascending=False).head(5)

        for _, row in weights_df.iterrows():
            records.append({
                'Scenario': scenario,
                'Shareholder': row['Name'],
                'Percentage': row['Percentage']
            })

    df_plot = pd.DataFrame(records)

    # Create anonymous labels
    unique_shareholders = df_plot['Shareholder'].unique()
    mapping = {name: f"Shareholder_{i + 1}" for i, name in enumerate(unique_shareholders)}
    df_plot['Anonymous_Label'] = df_plot['Shareholder'].map(mapping)

    # Pivot for plot
    label = 'Anonymous_Label' if anonymized else 'Shareholder'
    df_pivot = df_plot.pivot_table(index='Scenario', columns=label, values='Percentage')

    # Plot
    plt.figure(figsize=(10, 6))
    df_pivot.plot(kind='bar', figsize=(12, 6))

    plt.title('Top 5 Shareholders per Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Shares (%)')
    plt.legend(title='Shareholders', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return df_plot



def plot_mizuno_index(
    results,
    anonymized=True,
    target_name="Target",
    font_family='DejaVu Sans',
    font_size=14,
    linewidth=2,
    markersize=2,
    figsize=(32, 12)
):
    """Plots top 5 NPI and NPF values per scenario."""

    # Global font settings
    rcParams['font.family'] = font_family
    rcParams['font.size'] = font_size

    records_npi = []
    records_npf = []

    for scenario, data in results.items():
        npi_df = data['NPI_df'].copy()
        npf_df = data['NPF_df'].copy()

        # Top 5 non-zero values
        top_npi = npi_df[npi_df['NPI'] != 0].sort_values(by='NPI', ascending=False).head(5)
        top_npf = npf_df[npf_df['NPF'] != 0].sort_values(by='NPF', ascending=False).head(5)

        for _, row in top_npi.iterrows():
            records_npi.append({
                'Scenario': scenario,
                'Shareholder': row['Name'],
                'Value': row['NPI']
            })

        for _, row in top_npf.iterrows():
            records_npf.append({
                'Scenario': scenario,
                'Shareholder': row['Name'],
                'Value': row['NPF']
            })

    df_npi = pd.DataFrame(records_npi)
    df_npf = pd.DataFrame(records_npf)

    # Anonymous labels
    if anonymized:
        unique_names = pd.concat([df_npi['Shareholder'], df_npf['Shareholder']]).unique()
        mapping = {}
        counter = 1

        for name in unique_names:
            if name == target_name:
                mapping[name] = "Target"
            else:
                mapping[name] = f"Company_{counter}"
                counter += 1

        df_npi['Anonymous_Label'] = df_npi['Shareholder'].map(mapping)
        df_npf['Anonymous_Label'] = df_npf['Shareholder'].map(mapping)
        label_col = 'Anonymous_Label'
    else:
        label_col = 'Shareholder'

    pivot_npi = df_npi.pivot_table(index='Scenario', columns=label_col, values='Value').fillna(0)
    pivot_npf = df_npf.pivot_table(index='Scenario', columns=label_col, values='Value').fillna(0)

    # Color palette
    all_names = pd.Index(df_npi[label_col].unique()).union(df_npf[label_col].unique())
    cmap = cm.get_cmap('tab20', len(all_names))
    color_mapping = {name: cmap(i) for i, name in enumerate(all_names)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # NPI plot
    pivot_npi.plot(
        ax=axes[0],
        marker='o',
        color=[color_mapping.get(c) for c in pivot_npi.columns],
        linewidth=linewidth,
        markersize=markersize
    )
    axes[0].set_title('Top 5 NPI per Scenario', fontsize=font_size + 2)
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('NPI (%)')
    axes[0].legend(title='Companies', bbox_to_anchor=(1.05, 1), loc='upper left')

    # NPF plot
    pivot_npf.plot(
        ax=axes[1],
        marker='o',
        color=[color_mapping.get(c) for c in pivot_npf.columns],
        linewidth=linewidth,
        markersize=markersize
    )
    axes[1].set_title('Top 5 NPF per Scenario', fontsize=font_size + 2)
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('NPF (%)')
    axes[1].legend(title='Companies', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return df_npi, df_npf



# ---------------------------
# Functions for saving graphs with PyVis
# ---------------------------
def save_npf_graph_html_pyvis_anonymized(
    subgraph,
    npf_df,
    target=None,
    ultimate_owner=None,
    npi_uos=100,
    filename="html/npf_graph.html",
    node_size_base=20
):
    """
    Save an interactive PyVis graph of nodes with NPF > 0.
    Highlights target and ultimate owner and the path between them.
    Nodes are colored in shades of red based on NPF.
    """

    H = subgraph.copy()

    # Highlight paths
    highlighted_edges = set()
    if target and ultimate_owner:
        try:
            all_paths = list(nx.all_simple_paths(H, source=ultimate_owner, target=target))
            for path in all_paths:
                highlighted_edges.update(zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath:
            print(f"⚠️ No path between {ultimate_owner} and {target}")

    npf_dict = dict(zip(npf_df['ID'], npf_df['NPF']))
    max_npf = max(npf_dict.values()) if npf_dict else 1

    norm = Normalize(vmin=0, vmax=max_npf)
    cmap = cm.get_cmap("Reds")

    # Node styling
    for n in H.nodes():
        npf_val = npf_dict.get(n, 0)

        size = node_size_base if n == target else node_size_base + 60 * (npf_val / max_npf)
        H.nodes[n]['size'] = size

        if n == target:
            H.nodes[n]['color'] = "green"
        elif n == ultimate_owner:
            if npi_uos > 50:
                H.nodes[n]['color'] = "red"
            elif npi_uos > 25:
                H.nodes[n]['color'] = "orange"
            else:
                H.nodes[n]['color'] = "yellow"
        else:
            H.nodes[n]['color'] = to_hex(cmap(norm(npf_val)))

    # Edge styling
    for u, v in H.edges():
        H.edges[u, v]['color'] = "red" if (u, v) in highlighted_edges else "gray"
        H.edges[u, v]['width'] = 3 if (u, v) in highlighted_edges else 1.5

    net = Network(height="900px", width="100%", notebook=False, directed=True)

    for n in H.nodes():
        if n == target:
            label = "Target"
        elif n == ultimate_owner:
            label = "Ultimate Owner"
        else:
            label = " "
        net.add_node(
            n,
            label=label,
            color=H.nodes[n]['color'],
            size=H.nodes[n]['size'],
            font={"size": 40}
        )

    for u, v in H.edges():
        net.add_edge(u, v, color=H.edges[u, v]['color'], width=H.edges[u, v]['width'])

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    html_str = net.generate_html()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"Graph saved as '{filename}' ✅")
    return filename



def save_npf_graph_html_pyvis(
    subgraph,
    npf_df,
    target=None,
    ultimate_owner=None,
    npi_uos=100,
    filename="html/npf_graph.html",
    node_size_base=20
):
    """
    Save an interactive PyVis graph of nodes with NPF > 0.
    Labels enlarge dynamically on hover.
    """

    H = subgraph.copy()

    highlighted_edges = set()
    if target and ultimate_owner:
        try:
            all_paths = list(nx.all_simple_paths(H, source=ultimate_owner, target=target))
            for path in all_paths:
                highlighted_edges.update(zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath:
            print(f"⚠️ No path between {ultimate_owner} and {target}")

    npf_dict = dict(zip(npf_df['ID'], npf_df['NPF']))
    max_npf = max(npf_dict.values()) if npf_dict else 1

    norm = Normalize(vmin=0, vmax=max_npf)
    cmap = cm.get_cmap("Reds")

    # Node style
    for n in H.nodes():
        npf_val = npf_dict.get(n, 0)
        size = node_size_base if n == target else node_size_base + 60 * (npf_val / max_npf)
        H.nodes[n]['size'] = size

        if n == target:
            color = "green"
        elif n == ultimate_owner:
            if npi_uos > 50:
                color = "red"
            elif npi_uos > 25:
                color = "orange"
            else:
                color = "yellow"
        else:
            color = to_hex(cmap(norm(npf_val)))

        H.nodes[n]['color'] = color

    # Edge style
    for u, v in H.edges():
        H.edges[u, v]["color"] = "red" if (u, v) in highlighted_edges else "gray"
        H.edges[u, v]["width"] = 3 if (u, v) in highlighted_edges else 1.5

    net = Network(height="900px", width="100%", notebook=False, directed=True)

    for n in H.nodes():
        if n == target:
            label = "Target"
        elif n == ultimate_owner:
            label = "Ultimate Owner"
        else:
            label = H.nodes[n]['name']

        net.add_node(
            n,
            label=label,
            color=H.nodes[n]['color'],
            size=H.nodes[n]['size'],
            font={"size": 30}
        )

    for u, v in H.edges():
        net.add_edge(u, v, color=H.edges[u, v]['color'], width=H.edges[u, v]['width'])

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    html_str = net.generate_html()

    # JavaScript for label enlargement on hover
    js_hover = """
    <script type="text/javascript">
    network.on("hoverNode", function(params) {
        var nodeId = params.node;
        var options = network.body.nodes[nodeId].options;
        options.font.size = 60;
        network.redraw();
    });
    network.on("blurNode", function(params) {
        var nodeId = params.node;
        var options = network.body.nodes[nodeId].options;
        options.font.size = 30;
        network.redraw();
    });
    </script>
    """

    html_str = html_str.replace("</body>", js_hover + "\n</body>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"Graph saved as '{filename}' ✅ (hover effect enabled)")
    return filename
