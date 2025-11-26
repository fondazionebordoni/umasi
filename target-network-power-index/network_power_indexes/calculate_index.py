# ---------------------------
# Standard libraries
# ---------------------------
import time
import math
import cProfile
import pstats
from multiprocessing import cpu_count

# ---------------------------
# Libraries for data and numerical computation
# ---------------------------
import numpy as np
import pandas as pd
from rich import print

# ---------------------------
# Libraries for parallelization
# ---------------------------
from joblib import Parallel, delayed


def redistribute_value_all_leech(
        G_,
        weight_threshold=0,
        proportional=False,
        excluded_node=None,
        exclude_governments=False,
        leech_method=False,
        observation_threshold=0.25,
        majority_quota=50.0,
        verbose=False
):
    """
    Forces the total sum of incoming edges to 100 for each node, using various strategies:
    - equal or proportional distribution of missing values;
    - exclusion of governments from redistribution;
    - 'leech_method=True' applies Leech's two imputation cases:
        (C) Concentrated: all missing shares equal to the observation threshold;
        (D) Dispersed: “oceanic game” with modified control threshold.
    """

    G = G_.copy()

    for node in list(G.nodes):
        all_in_edges = [(u, v, d) for u, v, d in G.in_edges(node, data=True)]
        total_pct = sum(d.get("weight", 0) for _, _, d in all_in_edges)

        if math.isclose(total_pct, 100.0, abs_tol=1e-9):
            continue

        missing_value = 100.0 - total_pct

        # If Leech method is not active, use standard redistribution
        if not leech_method:
            eligible = []
            for u, v, d in all_in_edges:
                weight = d.get("weight", 0)
                if weight <= weight_threshold:
                    continue
                if excluded_node is not None and u == excluded_node:
                    continue
                if exclude_governments and "Government" in G.nodes[u].get("name", ""):
                    continue
                eligible.append((u, v, d))

            if len(eligible) == 0:
                continue

            eligible_sum = sum(d.get("weight", 0) for _, _, d in eligible)

            if not proportional:
                increment = missing_value / len(eligible)
                for _, _, d in eligible:
                    d["weight"] = d.get("weight", 0) + increment
            else:
                if math.isclose(eligible_sum, 0.0, abs_tol=1e-12):
                    increment = missing_value / len(eligible)
                    for _, _, d in eligible:
                        d["weight"] = d.get("weight", 0) + increment
                else:
                    for _, _, d in eligible:
                        current_weight = d.get("weight", 0)
                        share = current_weight / eligible_sum
                        d["weight"] = current_weight + missing_value * share

            # Numerical final correction
            new_total = sum(d.get("weight", 0) for _, _, d in all_in_edges)
            delta = 100.0 - new_total
            if abs(delta) > 1e-8:
                u0, v0, d0 = eligible[0]
                d0["weight"] += delta

        # --------------------------------------------------------------
        # Leech imputation method (Cases C and D)
        # --------------------------------------------------------------
        else:
            # 1. Determine observed shares (k) and total observed sum
            observed = [(u, v, d) for u, v, d in all_in_edges if d.get("weight", 0) > observation_threshold]
            s_k = sum(d.get("weight", 0) for _, _, d in observed) / 100.0  # observed fraction
            w_k = observation_threshold  # minimum observation threshold

            # 2. Estimated number of missing shareholders
            n_k = max(1, int((100.0 - s_k * 100.0) / w_k))

            # --- CASE (C): CONCENTRATED ---
            for u, v, d in observed:
                d["weight_C"] = d.get("weight", 0)

            # Create synthetic nodes for missing shareholders
            missing_weight_C = w_k
            for i in range(n_k):
                fake_node = f"Unobs_{i}_C"
                G.add_node(fake_node, type="unobserved")
                G.add_edge(fake_node, node, weight=missing_weight_C, weight_C=missing_weight_C)

            # --- CASE (D): DISPERSED (Oceanic game) ---
            q = majority_quota / 100.0
            q_mod = q - (1 - s_k) / 2.0  # modified control threshold

            for u, v, d in observed:
                d["weight_D"] = d.get("weight", 0)

            G.nodes[node]["modified_quota_D"] = q_mod

            if verbose:
                print(f"\nNode: {node}")
                print(f"  Leech method active.")
                print(f"  s_k (observed share): {s_k:.4f}")
                print(f"  Case (C): {n_k} synthetic participations of {w_k}% each.")
                print(f"  Case (D): modified majority threshold q' = {q_mod:.4f}")

    return G

def redistribute_value_all(
        G_,
        weight_threshold=0,
        proportional=False,
        excluded_node=None,
        exclude_governments=False,
        verbose=False
):
    """
    For each node in the graph, forces the total sum of its INCOMING edges to 100.
    - The amount to be redistributed (total_pct) considers ALL incoming edges.
    - Only "eligible" edges (above threshold, not excluded) receive the redistribution.
    - Redistribution can be: equal (proportional=False) or proportional to existing weights.
    - A final numerical correction ensures the total sums exactly to 100.
    """

    G = G_.copy()

    for node in G.nodes:

        # --- 1) compute the TOTAL sum of incoming edges (all of them) ---
        all_in_edges = [(u, v, d) for u, v, d in G.in_edges(node, data=True)]
        total_pct = sum(d.get("weight", 0) for _, _, d in all_in_edges)

        if verbose:
            print(f"\nNode: {node}")
            print(f"  Total sum (all incoming edges): {total_pct:.6f}")

        # If it's already 100 (within tolerance), continue
        if math.isclose(total_pct, 100.0, abs_tol=1e-9):
            if verbose:
                print("  ✅ Already = 100, nothing to do.")
            continue

        # Value to redistribute (positive if missing, negative if exceeding 100)
        missing_value = 100.0 - total_pct

        if verbose:
            print(f"  Value to redistribute (100 - total_pct): {missing_value:.6f}")

        # --- 2) Determine eligible edges ---
        eligible = []
        for u, v, d in all_in_edges:
            weight = d.get("weight", 0)

            # Eligible only if above threshold and not excluded
            if weight <= weight_threshold:
                continue
            if excluded_node is not None and u == excluded_node:
                continue
            if exclude_governments and "Government" in G.nodes[u].get("name", ""):
                continue
            eligible.append((u, v, d))

        if len(eligible) == 0:
            if verbose:
                print("  ⚠️ No eligible edges to redistribute the difference. No changes made.")
            continue

        # --- 3) Perform redistribution over eligible edges ---
        eligible_sum = sum(d.get("weight", 0) for _, _, d in eligible)

        if not proportional:
            # Equal redistribution
            increment = missing_value / len(eligible)
            for _, _, d in eligible:
                d["weight"] = d.get("weight", 0) + increment
        else:
            # Proportional redistribution
            if math.isclose(eligible_sum, 0.0, abs_tol=1e-12):
                increment = missing_value / len(eligible)
                for _, _, d in eligible:
                    d["weight"] = d.get("weight", 0) + increment
            else:
                for _, _, d in eligible:
                    current_weight = d.get("weight", 0)
                    share = current_weight / eligible_sum
                    d["weight"] = current_weight + missing_value * share

        # --- 4) Numerical correction to ensure sum = 100 exactly ---
        new_total = sum(d.get("weight", 0) for _, _, d in all_in_edges)
        new_total_rounded = round(new_total, 12)
        delta = 100.0 - new_total_rounded

        if abs(delta) > 1e-8:
            u0, v0, d0 = eligible[0]
            d0["weight"] = d0.get("weight", 0) + delta
            new_total = sum(d.get("weight", 0) for _, _, d in all_in_edges)

        if verbose:
            print(f"  Sum after modification (rounded): {round(new_total, 6):.6f}")
            if math.isclose(new_total, 100.0, abs_tol=1e-6):
                print("  ✅ Total corrected to 100.")
            else:
                print(f"  ⚠️ Total NOT = 100 (final value: {new_total:.12f}) — check data/filters.")

    return G



def scenario_chosen_all(scenario, subgraph, target):
    #### ======== Choose target company ========
    total_pct = round(sum(subgraph[u][v]['weight'] for u, v in subgraph.in_edges(target)), 2)
    weight_threshold = 1

    if scenario == '2':
        subgraph = redistribute_value_all(subgraph.copy(), weight_threshold=weight_threshold, proportional=True)
    elif scenario == '1':
        subgraph = redistribute_value_all(subgraph.copy(), weight_threshold=weight_threshold)
    elif scenario == '4':
        subgraph = redistribute_value_all(subgraph.copy(), weight_threshold=weight_threshold,
                                          proportional=True, exclude_governments=True)
    elif scenario == '3':
        subgraph = redistribute_value_all(subgraph.copy(), weight_threshold=weight_threshold,
                                          proportional=False, exclude_governments=True)

    print(f'You selected scenario {scenario}.')
    total_pct_2 = round(sum(subgraph[u][v]['weight'] for u, v in subgraph.in_edges(target)), 2)
    print(f'Total changed from {total_pct}% to {total_pct_2}%.')
    return subgraph



def convert_x(x):
    x_converted = []

    for entry in x:
        owners = list(entry.keys())
        shares = list(entry.values())
        x_converted.append({
            "owners": owners,
            "shares": shares
        })
    return x_converted


# ---------------------------
# Monte Carlo simulation block
# ---------------------------
def run_simulation_block(T_chunk, seed, x, v, labels, idx_E, q_j, d, init_prob, burn_in, steps, steps_rev):
    rng = np.random.default_rng(seed)
    n = len(labels)
    N = list(range(n))

    NPI_global = np.zeros(n)
    NPF_global = np.zeros(n)
    NPI_to_E = np.zeros(n)
    NPF_to_E = np.zeros(n)
    total_step = 0

    L_D = np.arange(n)
    L_I = np.arange(n)
    t_initialize = 0

    for t in range(1, T_chunk + 1):
        # Random reset / initialisation
        if t == 1 or rng.random() < init_prob:
            t_initialize = t
            L_D = np.arange(n)
            L_I = np.arange(n)
        else:
            for j in N:
                owners_j = x[j]["owners"]
                shares_j = x[j]["shares"]
                m = len(owners_j)
                U_seen = {}
                rows = []

                for k in range(m):
                    i = owners_j[k]
                    uo = L_I[i]

                    if uo not in U_seen:
                        U_seen[uo] = rng.random()

                    tag_uo = U_seen[uo]
                    tie = rng.random()

                    rows.append((tag_uo, tie, i))

                rows.sort(key=lambda tup: (tup[0], tup[1]))

                xsum = 0.0
                for (_, _, i_sorted) in rows:
                    pos = owners_j.index(i_sorted)
                    xsum += shares_j[pos]
                    if xsum >= q_j:
                        L_D[j] = i_sorted
                        L_I[j] = L_I[i_sorted]
                        break

        if (t - t_initialize) >= burn_in:
            total_step += 1

            # --------------------------- NPF global ---------------------------
            p = v.copy()
            p_next = np.zeros(n)
            for _ in range(steps):
                for j in N:
                    i = L_D[j]
                    if j != i:
                        p_next[i] += d * p[j]
                p_next += v
                p, p_next = p_next, np.zeros(n)
            NPF_global += p

            # --------------------------- NPI global ---------------------------
            for j in N:
                NPI_global[L_I[j]] += v[j]

            # --------------------------- NPI → target -------------------------
            uoE = L_I[idx_E]
            NPI_to_E[uoE] += v[idx_E]

            # --------------------------- NPF → target -------------------------
            children = [[] for _ in range(n)]
            for j in N:
                i = L_D[j]
                if j != i:
                    children[i].append(j)
            children[idx_E] = []

            for k in N:
                p_rev = np.zeros(n)
                p_rev[k] = v[k]
                p_next = np.zeros(n)
                acc_E = 0.0

                for _ in range(steps_rev):
                    acc_E += p_rev[idx_E]

                    for i in N:
                        if children[i]:
                            contrib = d * p_rev[i]
                            if contrib != 0.0:
                                for cj in children[i]:
                                    p_next[cj] += contrib

                    p_rev, p_next = p_next, np.zeros(n)

                NPF_to_E[k] += acc_E

    return NPI_global, NPF_global, NPI_to_E, NPF_to_E, total_step

def calculate_index(subgraph, target):
    """
    Main function computing influence/power indices over a given subgraph.
    """

    # ---------------------------
    # Random seed / setup
    # ---------------------------
    rng = np.random.default_rng(123)

    # ---------------------------
    # Input data
    # ---------------------------
    nodes = list(subgraph.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    N = list(range(n))

    # ---------------------------
    # Node labels
    # ---------------------------
    node_labels = [str(node) for node in nodes]
    labels = node_labels.copy()

    # Create ownership structure x
    x_sub = [None] * n

    for node in nodes:
        idx = idx_map[node]
        incoming_edges = subgraph.in_edges(node, data=True)

        if incoming_edges:
            # node with predecessors
            x_sub[idx] = {
                idx_map[src]: data["weight"] / 100.0
                for src, _, data in incoming_edges
            }
        else:
            # standalone node — 100% self–ownership
            x_sub[idx] = {idx: 1.0}

    # Convert to Mizuno-style structure
    x = convert_x(x_sub)

    # ---------------------------
    # Vector v (economic value)
    # ---------------------------
    v_raw = np.array([
        subgraph.nodes[n].get("value", 0)
        for n in nodes
    ], dtype=float)

    if np.all(v_raw == 0):
        print("Using uniform v")
        v = np.ones(n)
    else:
        print("Using economically scaled v")
        v_nonzero = v_raw[v_raw > 0]
        v_min, v_max = v_nonzero.min(), v_nonzero.max()
        v = np.array([
            0 if val == 0 else 0.1 + 0.9 * (val - v_min) / (v_max - v_min)
            for val in v_raw
        ])

    # ---------------------------
    # Parameters
    # ---------------------------
    total_perc = sum(subgraph[u][v]["weight"] for u, v in subgraph.in_edges(target)) / 100
    print(f"Total shareholding of {target} in the subgraph: {round(total_perc * 100, 2)}%")

    T = 20000                  # Monte Carlo iterations
    max_cores = 16
    n_jobs = min(cpu_count(), max_cores)
    T_chunk = T // n_jobs

    print(f"Using {n_jobs} cores, {T_chunk} iterations per core")
    q_j = total_perc / 2       # control threshold
    d = 0.85                   # damping factor
    init_prob = 0.02           # reset probability
    burn_in = 15
    steps = 50
    steps_rev = 50

    print(f"Running {T:,} Monte Carlo iterations across {n_jobs} processes...")

    # ---------------------------
    # Target label
    # ---------------------------
    target_label = target
    try:
        idx_target = labels.index(target_label)
    except ValueError:
        raise ValueError(f'Target "{target_label}" not found.')

    # Compute indices
    (
        NPI_global,
        NPF_global,
        NPI_to_target,
        NPF_to_target,
        total_step
    ) = mizuno_optimize(
        x, v, labels, idx_target,
        T_chunk, n,
        q_j, d, init_prob, burn_in,
        steps, steps_rev, n_jobs
    )

    (
        NPI_global,
        NPF_global,
        NPI_to_target,
        NPF_to_target
    ) = calculate_mean(
        NPI_global, NPF_global, NPI_to_target, NPF_to_target,
        x, v, N, n, total_step
    )

    return NPI_global, NPF_global, NPI_to_target, NPF_to_target, labels



def mizuno_optimize(
    x, v, labels, idx_target, T_chunk, n,
    q_j, d, init_prob, burn_in, steps, steps_rev,
    n_jobs
):
    """
    Parallelized Monte Carlo computation following Mizuno et al.
    """

    with cProfile.Profile() as pr:
        start = time.time()

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_simulation_block)(
                T_chunk, 123 + i, x, v, labels, idx_target,
                q_j, d, init_prob, burn_in, steps, steps_rev
            )
            for i in range(n_jobs)
        )

        print("Time:", time.time() - start)

    stats = pstats.Stats(pr)
    stats.sort_stats("tottime").print_stats(2)

    # Aggregate partial results
    NPI_global = np.zeros(n)
    NPF_global = np.zeros(n)
    NPI_to_target = np.zeros(n)
    NPF_to_target = np.zeros(n)
    total_step = 0

    for r in results:
        NPI_global += r[0]
        NPF_global += r[1]
        NPI_to_target += r[2]
        NPF_to_target += r[3]
        total_step += r[4]

    return NPI_global, NPF_global, NPI_to_target, NPF_to_target, total_step



def calculate_mean(
    NPI_global, NPF_global, NPI_to_target, NPF_to_target,
    x, v, N, n, total_step
):
    """
    Normalize indices across valid iterations.
    """

    if total_step == 0:
        raise RuntimeError("No valid iteration after burn-in.")

    NPI_global /= total_step
    NPF_global /= total_step
    NPI_to_target /= total_step
    NPF_to_target /= total_step

    # Normalize global NPF (simple rho)
    self_full = np.zeros(n, dtype=bool)
    for j in N:
        owners_j = x[j]["owners"]
        shares_j = x[j]["shares"]
        try:
            pos = owners_j.index(j)
            if abs(shares_j[pos] - 1.0) < 1e-12:
                self_full[j] = True
        except ValueError:
            pass

    denom = NPF_global[self_full].sum()
    rho = (v.sum() / denom) if denom > 0 else 1.0
    NPF_global = NPF_global * rho

    return NPI_global, NPF_global, NPI_to_target, NPF_to_target



def calculate_shareholdings(subgraph, target, verbose=True):
    """
    Computes direct shareholding percentages toward the target.
    """

    weights = []
    nodes = []

    for node in subgraph.nodes:
        if node != target:
            if subgraph.has_edge(node, target):
                w = subgraph.get_edge_data(node, target)["weight"]
                if verbose:
                    print(f"Edge {node}→{target} weight: {round(w, 2)}%")
            else:
                w = 0
        else:
            w = None

        nodes.append(node)
        weights.append(w)

    df = pd.DataFrame({"PermID": nodes, "Percentage": weights})
    return df



def save_index(NPX_to_target, labels, subgraph, share_df,
               target, company_name, scenario, index="NPI",
               verbose=False):
    """
    Export top influencers + shareholdings to Excel.
    """

    df_index = pd.DataFrame([NPX_to_target], columns=labels)
    top10 = df_index.T.sort_values(by=0, ascending=False)

    if verbose:
        print(f"\n{index} → {company_name}")

    ids, names, values, perc = [], [], [], []

    for node_id in top10.index:
        name = subgraph.nodes[node_id]["name"]

        if verbose:
            print(node_id)
            print(f"\t[bold]{name}[/bold]")
            if index == "NPI":
                msg = f"\t\t{round(top10.loc[node_id][0] * 100, 2)}% (Control probability)"
            else:
                msg = f"\t\t{round(top10.loc[node_id][0] * 100, 2)}% (Node importance)"
            print(msg)
            print(
                f'\t\t\tShareholding: {round(share_df.loc[share_df["PermID"] == node_id, "Percentage"].iloc[0], 2)}%'
            )
            print()

        ids.append(node_id)
        names.append(name)
        values.append(top10.loc[node_id][0])
        perc.append(share_df.loc[share_df["PermID"] == node_id, "Percentage"].iloc[0])

    out_df = pd.DataFrame({
        "ID": ids,
        "Name": names,
        index: values,
        "Shareholding": perc
    })

    if index == "NPI":
        print(
            f"\n\nUltimate Owner of {subgraph.nodes[target]['name']} is "
            f"[bold]{subgraph.nodes[top10.iloc[0].name]['name']}[/bold] "
            f"(NPI = {round(top10.iloc[0][0] * 100, 2)}%)"
        )

    out_df.to_excel(f"excel/{index}_{company_name}_{scenario}.xlsx")
    return out_df



def add_index_to_graph(npi_df, npf_df, graph, default_value=0.0):
    """
    Adds NPI and NPF values to the graph nodes.
    """

    subgraph = graph.copy()

    name_to_node = {
        data.get("name"): node
        for node, data in subgraph.nodes(data=True)
        if "name" in data
    }

    npi_map = dict(zip(npi_df["Name"], npi_df["NPI"]))
    npf_map = dict(zip(npf_df["Name"], npf_df["NPF"]))

    for node, data in subgraph.nodes(data=True):
        name = data.get("name")

        npi_value = npi_map.get(name, default_value)
        npf_value = npf_map.get(name, default_value)

        subgraph.nodes[node]["NPI"] = float(npi_value)
        subgraph.nodes[node]["NPF"] = float(npf_value)

    return subgraph



def add_index_to_graph_old(npi_df, npf_df, graph):
    """
    Legacy version of add_index_to_graph.
    """

    subgraph = graph.copy()
    name_to_node = {d["name"]: n for n, d in subgraph.nodes(data=True)}

    for _, row in npi_df.iterrows():
        n = name_to_node.get(row["Name"])
        if n is not None:
            subgraph.nodes[n]["NPI"] = row["NPI"] * 100
            subgraph.nodes[n]["NPF"] = npf_df.loc[
                npf_df["Name"] == row["Name"], "NPF"
            ].values[0] * 100

    return subgraph
