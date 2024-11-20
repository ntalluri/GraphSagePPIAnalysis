# import tarfile

# # Path to the tar.gz file
# file_path = 'bio-tissue-networks.tar.gz'

# # Open and extract the contents of the file
# with tarfile.open(file_path, 'r:gz') as tar:
#     tar.extractall()  # Extracts to the current directory
#     print("Extraction completed.")


import networkx as nx
import json
import pandas as pd
import os

os.makedirs("data/graphs/graphml", exist_ok=True)
os.makedirs("data/graphs/edge_lists", exist_ok=True)

with open('data/ppi/ppi-G.json', 'r') as file:
    ppi_graph = json.load(file)

print(ppi_graph.keys())

G = nx.Graph()
for node in ppi_graph['nodes']:
    node_id = node.get('id')
    G.add_node(node_id, **node)

for link in ppi_graph['links']:
    source = link.get('source')
    target = link.get('target')
    G.add_edge(source, target, **link)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is the graph connected?", nx.is_connected(G))


num_components = nx.number_connected_components(G)
print("Number of connected components:", num_components)

components = list(nx.connected_components(G))

small_components = [c for c in components if len(c) < 100]

# Remove nodes in small components from the graph
for component in small_components:
    G.remove_nodes_from(component)

num_components = nx.number_connected_components(G)
print("Number of connected components:", num_components)

nx.write_graphml(G, "data/ppi_network.graphml")


components = list(nx.connected_components(G))

for i, component_nodes in enumerate(components):
    # Create a subgraph for the component
    component_subgraph = G.subgraph(component_nodes).copy()
    
    # Save the subgraph to a file
    graphml_filename = f"data/graphs/graphml/component_{i}.graphml"
    nx.write_graphml(component_subgraph, graphml_filename)
    edgelist_filename = f"data/graphs/edge_lists/component_{i}.edgelist"
    nx.write_edgelist(component_subgraph, edgelist_filename, data=False, delimiter="\t")

