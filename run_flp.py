import csv
import os
import random
import pandas as pd
import networkx as nx
from pathlib import Path
from networkx.algorithms.community import modularity
from time import perf_counter
from flp import flp

def graph_processing(file_name: str, data_name: str):
    graph_path = f"{file_name}/{data_name}/edges.txt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    df_edges = pd.read_csv(graph_path, sep=r"\s+", header=None, names=["source", "target"])
    graph = nx.Graph()
    graph.add_edges_from(df_edges.values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    feature_path = f"{file_name}/{data_name}/attributes.txt"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
    df_space = pd.read_csv(feature_path, sep=r"\s+", header=None, names=["node", "attribute"], engine="python")
    if df_space["attribute"].isnull().any():
        df_comma = pd.read_csv(feature_path, delimiter=",", header=None, names=["node", "attribute"])
        df_attr = df_comma
    else:
        df_attr = df_space
        
    group_membership = dict(zip(df_attr["node"], df_attr["attribute"]))
    group_membership = {node: attr for node, attr in group_membership.items() if node in graph}
    nx.set_node_attributes(graph, group_membership, "protected")
    
    nodes_to_remove = [node for node, data in graph.nodes(data=True) if "protected" not in data]
    graph.remove_nodes_from(nodes_to_remove)
    largest_component_nodes = max(nx.connected_components(graph), key=len)
    
    return graph.subgraph(largest_component_nodes).copy()

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

def init_df_balance(communities, G, attribute="protected"):
    rows = []
    attrs = nx.get_node_attributes(G, attribute)
    allred = sum(1 for v in attrs.values() if v == 1)
    allblue = sum(1 for v in attrs.values() if v != 1)
    global_group_ratios = {1: allblue / allred, 0: allred / allblue}
    
    for com_id, nodes in enumerate(communities):
        red_count = sum(1 for n in nodes if attrs[n] == 1)
        blue_count = len(nodes) - red_count
        size = len(nodes)
        
        balance = min(red_count / blue_count, blue_count / red_count) if red_count > 0 and blue_count > 0 else 0
        majority_group = 1 if red_count > blue_count else 0

        if balance == 1:
            max_value = max(global_group_ratios.values())
            normalized_balance = balance / max_value
        else:
            global_ratio = global_group_ratios[majority_group]
            normalized_balance = balance / global_ratio if global_ratio != 0 else 0

        rows.append({
            "community": com_id, "red": red_count, "blue": blue_count,
            "size": size, "balance": balance, "normalized_balance": normalized_balance,
            "more_red": majority_group
        })
    return pd.DataFrame(rows)

def make_df_results_lp(G, communities, results_df, type_Kc, flag):
    mod = modularity(G, communities)
    df = init_df_balance(communities, G, "protected")
    weighted_balance, globalbalance, unweighted_balance, avg_newblc = calculate_balances(df)

    algo_name = f"LP{type_Kc}" if type_Kc in [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9] else ("LP" if type_Kc == 0 else type_Kc)
    
    return pd.DataFrame([[
        algo_name, len(communities), unweighted_balance, weighted_balance,
        mod, globalbalance, avg_newblc, flag
    ]], columns=[
        "Algorithm", "Number of Communities", "Balance (Unweighted)", "Weighted Balance",
        "Modularity", "Global Balance", "Normalized Balance", "flag"
    ])

def evaluate_dataset(dataset, dataset_folder, k_g, k_c, max_iter):
    save_dir = f"flp_results/{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    G = graph_processing(dataset_folder, dataset)
    
    all_iterations_results = []
    
    for i in range(max_iter):
        print(f"Dataset {dataset} Iteration {i+1}/{max_iter}...")
        seed = random.randint(0, 10000)
        
        start_time = perf_counter()
        output_coms, results_df, flag = flp(G, k_g, k_c, seed=seed)
        end_time = perf_counter()
        
        with open(f"{save_dir}/execution_times.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if i == 0 and not os.path.exists(f"{save_dir}/execution_times.csv"):
                writer.writerow(["Algorithm", "Duration (seconds)"])
            writer.writerow([f"FLP{k_c}", end_time - start_time])
            
        metrics_df = make_df_results_lp(G, list(output_coms), results_df, k_c, flag)
        all_iterations_results.append(metrics_df)
        
    final_combined_df = pd.concat(all_iterations_results, ignore_index=True)
    results_file = os.path.join(save_dir, "results.csv")
    
    if os.path.exists(results_file):
        final_combined_df.to_csv(results_file, mode="a", index=False, header=False)
    else:
        final_combined_df.to_csv(results_file, index=False)


if __name__ == "__main__":
    datasets = ["friendship_net", "fb_ego", "deezer_europe", "filtered_drug_net", "filtered_twitter"]
    dataset_folder = "datasets"
    
    max_iter = 10
    k_c = 0.5 # lambda 
    k_g = 1- k_c # 1-lambda 

    for dataset in datasets:
        evaluate_dataset(dataset, dataset_folder, k_g, k_c, max_iter)
