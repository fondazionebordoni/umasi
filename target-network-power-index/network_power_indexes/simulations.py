import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import networkx as nx

def plot_share_and_npi(results, companies_analyzed, uo1_name, uo2_name, font_size=16):
    """
    Generates two side-by-side plots:
      1. Shareholding percentages of shareholders (anonymized)
      2. NPI of two ultimate owners (anonymized)

    Args:
        results (dict): Dictionary of simulation results, key = share_level
                        Values: {"weights_df": pd.DataFrame, "NPI_df": pd.DataFrame}
        companies_analyzed (list): List of shareholder names to anonymize
        uo1_name (str): Real name of the first ultimate owner
        uo2_name (str): Real name of the second ultimate owner
        font_size (int, optional): Font size for plots. Default = 16
    """

    # --------- Map real names → anonymous ---------
    anon_map = {name: f"SH{i+1}" for i, name in enumerate(companies_analyzed)}

    # Anonymize ultimate owners
    uo1_anon = "UO1"
    uo2_anon = "UO2"
    anon_map[uo1_name] = uo1_anon
    anon_map[uo2_name] = uo2_anon

    # --------- Sort simulations by start_share ---------
    sorted_keys = sorted(results.keys(), key=lambda x: float(x))

    # --------- Lists for plotting ---------
    percent_uo1 = []
    percent_shareholders = {anon_map[name]: [] for name in companies_analyzed}
    npi_uo1 = []
    npi_uo2 = []

    # --------- Fill lists ---------
    for key in sorted_keys:
        res = results[key]
        weights_df = res["weight_df"]
        npi_df = res["NPI_df"]

        # Percentages of UO1
        df_uo1 = weights_df[weights_df["Names"] == uo1_name]
        percent_uo1.append(df_uo1.iloc[0]["Percentage"] if len(df_uo1) > 0 else 0)

        # Percentages of other shareholders
        for real_name in companies_analyzed:
            anon_name = anon_map[real_name]
            df_sh = weights_df[weights_df["Names"] == real_name]
            percent_shareholders[anon_name].append(df_sh.iloc[0]["Percentage"] if len(df_sh) > 0 else 0)

        # NPI values for UO1 and UO2
        row1 = npi_df[npi_df["Name"] == uo1_name]
        npi_uo1.append(row1.iloc[0]["NPI"] if len(row1) > 0 else None)

        row2 = npi_df[npi_df["Name"] == uo2_name]
        npi_uo2.append(row2.iloc[0]["NPI"] if len(row2) > 0 else None)

    # =========================
    #         PLOTS
    # =========================
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # --------- Plot 1: Share Percentages ---------
    for anon_sh, vals in percent_shareholders.items():
        ax[0].plot(sorted_keys, vals, marker='o', label=anon_sh)

    ax[0].plot(sorted_keys, percent_uo1, marker='o', linewidth=4, color='black', label=uo1_anon)
    ax[0].set_title("Shareholding Percentages (Anonymized)", fontsize=22)
    ax[0].set_xlabel("Simulation", fontsize=18)
    ax[0].set_ylabel("Percentage %", fontsize=18)
    ax[0].legend(fontsize=18)
    ax[0].grid(True)

    # --------- Plot 2: NPI ---------
    ax[1].plot(sorted_keys, npi_uo1, marker='o', linewidth=4, color='black', label=uo1_anon)
    ax[1].plot(sorted_keys, npi_uo2, marker='o', label=uo2_anon)
    ax[1].set_title("NPI of UO1 vs UO2 (Anonymized)", fontsize=22)
    ax[1].set_xlabel("Simulation", fontsize=18)
    ax[1].set_ylabel("NPI", fontsize=18)
    ax[1].legend(fontsize=18)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_ultimate_owner_evolution(all_results, target_company, font_size=26):
    """
    Generates a plot showing the evolution of main shareholders' stakes
    simulating different ownership levels for each company.

    Args:
        all_results (dict): Dictionary of simulation results, structured as:
            {company: {share_level: {'NPI_df': pd.DataFrame}}}
        target_company (str): Name of the target company
        font_size (int): Font size for plot text
    """

    # Global plot parameters
    plt.rcParams.update({
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size
    })

    # Build summary DataFrame
    records_owner = []

    for company, sim_results in all_results.items():
        for share_level, data in sim_results.items():
            npi_df = data['NPI_df']

            owner = npi_df.iloc[0]['Name']
            share = npi_df.iloc[0]['Shareholding']
            npi = npi_df.iloc[0]['NPI']

            records_owner.append({
                'Company': company,
                'Simulated_share': float(share_level),
                'Ultimate_owner': owner,
                'Owner_share': share,
                'Owner_NPI': npi
            })

    df_owner = pd.DataFrame(records_owner)
    df_owner = df_owner.sort_values(by=['Company', 'Simulated_share'])

    # Consistent anonymous labels
    df_plot_npf = pd.read_excel(f'excel/NPF_{target_company}_2.xlsx')

    unique_names = df_plot_npf['Name'].dropna().unique()
    mapping = {}
    counter = 1
    for name in unique_names:
        if name == target_company:
            mapping[name] = "Target"
        else:
            mapping[name] = f"Company_{counter}"
            counter += 1

    df_plot_npf['Anon_label'] = df_plot_npf['Name'].map(mapping)
    anon_mapping = df_plot_npf.set_index("Name")["Anon_label"].to_dict()

    missing_names = set(df_owner['Company']).union(df_owner['Ultimate_owner']) - set(anon_mapping.keys())
    for i, name in enumerate(sorted(missing_names), start=len(anon_mapping) + 1):
        anon_mapping[name] = f"Entity_{i}"

    df_owner['Anon_company'] = df_owner['Company'].map(anon_mapping)
    df_owner['Anon_owner'] = df_owner['Ultimate_owner'].map(anon_mapping)

    # Setup colors and subplots
    anon_companies = df_owner['Anon_company'].unique()
    n_companies = len(anon_companies)
    colors = cm.get_cmap('tab10', n_companies)

    n_cols = 2
    n_rows = math.ceil(n_companies / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 7 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot for each company
    for i, (ax, anon_company) in enumerate(zip(axes, anon_companies)):
        subset = df_owner[df_owner['Anon_company'] == anon_company]
        color = colors(i / n_companies)

        ax.plot(subset['Simulated_share'], subset['Owner_share'],
                marker='o', linestyle='-', color=color, label=anon_company,
                linewidth=5, markersize=20)

        for _, row in subset.iterrows():
            ax.text(row['Simulated_share'], row['Owner_share'] + 0.8, row['Anon_owner'],
                    fontsize=font_size, rotation=15, ha='left', va='bottom', color=color)

        ax.set_title(f"Ultimate Owner Evolution — {anon_company}", color=color)
        ax.set_xlabel("Simulated Share (%)")
        ax.set_ylabel("Ultimate Owner Share (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2, w_pad=3)
    plt.show()

def find_interconnected_groups(G, target):
    """
    Finds groups of shareholders that are interconnected among themselves without going through the target.

    Args:
        G (nx.DiGraph): Directed graph with ownership relationships
        target (str/int): Target node (controlled company)

    Returns:
        groups (list[set]): List of sets with IDs of interconnected nodes
        labels (dict): Mapping node_id -> "id (name)"
    """

    # --------------- Readable label ---------------
    def node_label(G, n):
        return f"{n} ({G.nodes[n].get('name', 'no-name')})"

    # --------------- Find direct shareholders ---------------
    shareholders = list(G.predecessors(target))

    # --------------- Undirected version of the graph ---------------
    H = G.to_undirected().copy()

    # Remove the target to avoid paths that go through it
    if target in H:
        H.remove_node(target)

    # --------------- Find interconnected groups ---------------
    interconnected_groups = []

    for i in range(len(shareholders)):
        for j in range(i+1, len(shareholders)):
            a, b = shareholders[i], shareholders[j]

            if nx.has_path(H, a, b):
                found_group = None

                # Check if a or b belong to an existing group
                for g in interconnected_groups:
                    if a in g or b in g:
                        found_group = g
                        break

                # Update existing group or create a new one
                if found_group:
                    found_group.update([a, b])
                else:
                    interconnected_groups.append(set([a, b]))

    # --------------- Build readable labels ---------------
    labels = {n: node_label(G, n) for n in shareholders}

    print("Interconnected groups:")
    for i, g in enumerate(interconnected_groups, 1):
        print(f"\nGroup {i}:")
        for n in g:
            print("  -", labels[n])

    # If you want the indices of the first group:
    indices = list(interconnected_groups[0]) if interconnected_groups else []

    return indices

def change_shares(G, df, target, shareholder_sim, new_pct):

    # 🔹 Copy original graph
    graph = G.copy()
    df_shareholders = df.copy()

    # 🔹 Find node ID corresponding to the shareholder name
    shareholder_id = df_shareholders.loc[df_shareholders['Name'] == shareholder_sim, 'ID'].values
    if len(shareholder_id) == 0:
        raise ValueError(f"Shareholder {shareholder_sim} not found among direct shareholders")
    shareholder_id = shareholder_id[0]

    # 🔹 Update edge weight towards target
    graph[shareholder_id][target]['weight'] = new_pct

    # 🔹 (Optional) Update DataFrame as well
    df_shareholders.loc[df_shareholders['ID'] == shareholder_id, 'Percentage'] = new_pct

    print(f"Updated {shareholder_sim} ({shareholder_id}) to {new_pct}% on target {target}")

    return graph, df_shareholders


def change_multishares(G, df, target, shareholders_sim, new_pcts):

    # 🔹 Copy graph and DataFrame
    graph = G.copy()
    df_shareholders = df.copy()

    # 🔹 If new_pcts is a list → convert to dict
    if isinstance(new_pcts, list):
        if len(new_pcts) != len(shareholders_sim):
            raise ValueError("The list of new percentages must match the number of shareholders.")
        new_pcts = {sh: p for sh, p in zip(shareholders_sim, new_pcts)}

    # 🔹 Loop over all shareholders to update
    for shareholder in shareholders_sim:

        if shareholder not in new_pcts:
            raise ValueError(f"Missing new percentage for {shareholder}")

        new_pct = new_pcts[shareholder]

        # 🔹 Find shareholder ID
        shareholder_id = df_shareholders.loc[df_shareholders["Name"] == shareholder, "ID"].values
        if len(shareholder_id) == 0:
            raise ValueError(f"Shareholder {shareholder} not found among direct shareholders")

        shareholder_id = shareholder_id[0]

        # 🔹 Update edge weight in graph
        graph[shareholder_id][target]["weight"] = new_pct

        # 🔹 Update DataFrame as well
        df_shareholders.loc[df_shareholders["ID"] == shareholder_id, "Percentage"] = new_pct

        print(f"Updated {shareholder} ({shareholder_id}) to {new_pct}% on target {target}")

    return graph, df_shareholders
