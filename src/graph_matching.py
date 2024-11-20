import networkx as nx
import pandas as pd
import os
from pathlib import Path

def load_graph(file_path):
    return nx.read_edgelist(file_path, delimiter="\t")

    
def compare_graphs(graph1, graph2):
    # compare nodes
    # if graph1.number_of_nodes() != graph2.number_of_nodes():
    #     return False

    # compare edges
    # if graph1.number_of_edges() != graph2.number_of_edges():
    #     return False
    
    # compare degree
    # degree_seq1 = sorted([d for _, d in graph1.degree()])
    # degree_seq2 = sorted([d for _, d in graph2.degree()])
    # if degree_seq1 != degree_seq2:
    #     return False

    # compare graph structure
    if not nx.is_isomorphic(graph1, graph2):
        return False

    return True

def find_matching_graphs(folder1, folder2):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    match_dict = {}

    graph2_dict = {}
    print("Loading graph2 files...")
    for graph_file2 in folder2.glob("*.edgelist"):
        graph2 = load_graph(graph_file2)
        if len(graph2.edges()) <= 4:
            continue  # Skip graphs with 4 or fewer edges
        graph2_dict[graph_file2.name] = graph2
        print(f"Processed {graph_file2.name}")
    print(f"Loaded {len(graph2_dict)} graph2 files.")


    for graph_file1 in folder1.glob("*.edgelist"): 
        graph1 = load_graph(graph_file1)
        print(f"Processed {graph_file1.name}")
        match_dict[graph_file1.name] = [] 
        for graph2_name, graph2 in graph2_dict.items():
            if compare_graphs(graph1, graph2):
                print(f"{graph_file1.name} and {graph2_name} match")
                match_dict[graph_file1.name].append(graph2_name)
        
    return match_dict

# Example usage
folder1_path = "data/bio-tissue-networks"
folder2_path = "data/graphs/edge_lists"
matches = find_matching_graphs(folder1_path, folder2_path)

with open("graph_matches.txt", "w") as f:
    for graph1, matching_graphs in matches.items():
        match_str = ", ".join(matching_graphs) if matching_graphs else "No matches"
        f.write(f"{graph1} matches with: {match_str}\n")

print("Matching graphs saved to graph_matches.txt")
