from collections import defaultdict
import csv
import os
import random
import pandas as pd
import networkx as nx
from pathlib import Path
from networkx.algorithms.community import modularity
from time import perf_counter
from flp import flp


def load_graph(file_name: str, data_name: str):

    graph_path = f"{file_name}/{data_name}/edges.txt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    df = pd.read_csv(graph_path, sep=r"\s+", header=None, names=["source", "target"])
    graph = nx.Graph()
    graph.add_edges_from(df.values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def graph_processing(file_name: str, data_name: str):

    graph = load_graph(file_name, data_name)
    print("load Number of nodes:", graph.number_of_nodes())
    print("load Number of edges:", graph.number_of_edges())
    feature_path = f"{file_name}/{data_name}/attributes.txt"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    df_space = pd.read_csv(feature_path, sep=r"\s+", header=None, names=["node", "attribute"], engine="python")
    if df_space["attribute"].isnull().any():
        df_comma = pd.read_csv(feature_path, delimiter=",", header=None, names=["node", "attribute"])
        if df_comma["attribute"].isnull().any():
            raise ValueError("Feature file delimiter could not be determined")
        df = df_comma
    else:
        df = df_space
    group_membership = dict(zip(df["node"], df["attribute"]))
    group_membership = {node: attr for node, attr in group_membership.items() if node in graph}
    nx.set_node_attributes(graph, group_membership, "protected")
    nodes_to_remove = [node for node, data in graph.nodes(data=True) if "protected" not in data]
    graph.remove_nodes_from(nodes_to_remove)
    largest_component_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_component_nodes).copy()
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    return graph


def savetime(folder_path, start, end, algorithm):

    time_file = Path(folder_path) / "execution_times.csv"
    duration = end - start
    time_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not time_file.exists()
    with open(time_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Algorithm", "Duration (seconds)"])
        writer.writerow([algorithm, duration])


def calculate_balances(df):

    zeroCounts = df["blue"].sum()
    oneCounts = df["red"].sum()
    com_balance = df["balance"]
    nodesINcom = df["size"]
    normalized_balance = df["normalized_balance"]
    avg_newblc = normalized_balance.mean()
    global_balance = min(zeroCounts / oneCounts, oneCounts / zeroCounts)
    weighted_avg_bal = (com_balance * nodesINcom).sum() / nodesINcom.sum()
    unweighted_avg = com_balance.mean()

    return weighted_avg_bal, global_balance, unweighted_avg, avg_newblc


def init_df_balance(communities, G, attribute):

    rows = []
    attrs = nx.get_node_attributes(G, attribute)
    allred = sum(1 for v in attrs.values() if v == 1)
    allblue = sum(1 for v in attrs.values() if v != 1)
    global_group_ratios = {
        1: allblue / allred,
        0: allred / allblue
    }
    for com_id, nodes in enumerate(communities):

        red_count = sum(1 for n in nodes if attrs[n] == 1)
        blue_count = len(nodes) - red_count
        size = len(nodes)
        if red_count > 0 and blue_count > 0:
            balance = min(red_count / blue_count, blue_count / red_count)
        else:
            balance = 0
        majority_group = 1 if red_count > blue_count else 0

        if balance == 1:
            max_value = max(global_group_ratios.values())
            normalized_balance = balance / max_value
        else:
            global_ratio = global_group_ratios[majority_group]
            normalized_balance = balance / global_ratio if global_ratio != 0 else 0

        rows.append({
            "community": com_id,
            "red": red_count,
            "blue": blue_count,
            "size": size,
            "balance": balance,
            "normalized_balance": normalized_balance,
            "more_red": majority_group
        })
    df = pd.DataFrame(rows)
    return df


def make_df_results_lp(G, communities, results_df, type_Kc, flag):

    lencom = len(communities)
    df = results_df
    mod = modularity(G, communities)

    if "normalized_balance" in df.columns:
        weighted_balance, globalbalance, unweighted_balance, avg_newblc = calculate_balances(df)
    else:
        df = init_df_balance(communities, G, "protected")
        weighted_balance, globalbalance, unweighted_balance, avg_newblc = calculate_balances(df)

    colums_names_algo = {
        0: "LP", 0.5: "LP05", 0.1: "LP01", 0.2: "LP02", 0.3: "LP03",
        0.4: "LP04", 0.6: "LP06", 0.7: "LP07", 0.8: "LP08",
        0.9: "LP09", 1: "LP1"
    }
    algo_name = colums_names_algo.get(type_Kc, type_Kc)
    results = pd.DataFrame([[
        algo_name,
        lencom,
        unweighted_balance,
        weighted_balance,
        mod,
        globalbalance,
        avg_newblc,
        flag
    ]], columns=[
        "Algorithm",
        "Number of Communities",
        "Balance",
        "Weighted Balance",
        "Modularity",
        "Global Balance",
        "Normalized Balance",
        "flag"
    ])

    return results

def run_algo_with_stats(G, k_g, k_c, folder_path, seed=42):

    start = perf_counter()
    output_coms, results_df, flag = flp(G, k_g, k_c, seed=seed)
    end = perf_counter()
    savetime(folder_path, start, end, f"FLP{k_c}")
    communities = list(output_coms)
    results_df = make_df_results_lp(G, communities, results_df, k_c, flag)
    return results_df

def run_algorithm(dataset, dataset_folder, k_g, k_c):

    df_path = f"flp_results/{dataset}"
    os.makedirs(df_path, exist_ok=True)
    combined_file = os.path.join(df_path, "all_dfs.csv")
    G = graph_processing(dataset_folder, dataset)
    dfs = []
    seed = random.randint(0, 10000)
    results_flp = run_algo_with_stats(G, k_g, k_c, df_path, seed=seed)
    results_flp["type_of_algorithm"] = "FLP"
    dfs.append(results_flp)
    new_combined_df = pd.concat(dfs, ignore_index=True)
    if os.path.exists(combined_file):
        new_combined_df.to_csv(combined_file, mode="a", index=False, header=False)
    else:
        new_combined_df.to_csv(combined_file, index=False)


if __name__ == "__main__":

    datasets = [
        "friendship_net",
        "fb_ego",
        "deezer_europe",
        "filtered_drug_net",
        "filtered_twitter"
    ]
    dataset_folder = "datasets"
    max_iter = 10
    k_c = 0.5
    k_g = 0.5

    for dataset in datasets:
        for i in range(max_iter):
            run_algorithm(dataset, dataset_folder, k_g, k_c)