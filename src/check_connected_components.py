import networkx as nx
import pandas as pd
import os
from pathlib import Path
import numpy as np

# import tarfile

# # Path to the tar.gz file
# file_path = 'bio-tissue-networks.tar.gz'

# # Open and extract the contents of the file
# with tarfile.open(file_path, 'r:gz') as tar:
#     tar.extractall()  # Extracts to the current directory
#     print("Extraction completed.")


def load_graph(file_path):
    return nx.read_edgelist(file_path, delimiter="\t")

folder1_path = "data/bio-tissue-networks"
folder2_path = "data/bio-tissue-networks-clean" 
folder3_path = "data/graphs/edge_lists"

# check that the bio tissue networks are connected, otherwise save the largest component part of the network
folder1 = Path(folder1_path)
for graph_file1 in folder1.glob("*.edgelist"): 
    graph1 = load_graph(graph_file1)
    print(graph_file1)
    print("Is the graph connected?", nx.is_connected(graph1))

    if not nx.is_connected(graph1):
        components = list(nx.connected_components(graph1))
        for i, component_nodes in enumerate(components):
            component_subgraph = graph1.subgraph(component_nodes).copy()
            if component_subgraph.number_of_edges() > 5:
                name = str(graph_file1).split("/")[2]
                print(name)
                edgelist_filename = f"data/bio-tissue-networks-clean/{name}"
                nx.write_edgelist(component_subgraph, edgelist_filename, data=False, delimiter="\t")

# checking the bio tissue networks are connected after "cleaning"
folder2 = Path(folder2_path)
for graph_file2 in folder2.glob("*.edgelist"): 
    graph2 = load_graph(graph_file2)
    print(graph_file2)
    print("Is the graph connected?", nx.is_connected(graph2))

# check that the components graphs from graphsage are fully connected
folder3 = Path(folder3_path)
for graph_file3 in folder3.glob("*.edgelist"):
    print(graph_file3)
    graph3 = load_graph(graph_file3)
    print("Is the graph connected?", nx.is_connected(graph3))

