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

with open('ppi/ppi-G.json', 'r') as file:
    ppi_graph = json.load(file)

print(ppi_graph.keys())

G = nx.Graph()
# Add nodes with attributes if present
for node in ppi_graph['nodes']:
    node_id = node.get('id')  # Assuming each node has an 'id' field
    G.add_node(node_id, **node)  # Add all node attributes as a dictionary

# Add edges
for link in ppi_graph['links']:
    source = link.get('source')  # Assuming 'source' field for starting node
    target = link.get('target')  # Assuming 'target' field for ending node
    G.add_edge(source, target, **link)  # Add all edge attributes as a dictionary


print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is the graph connected?", nx.is_connected(G))


num_components = nx.number_connected_components(G)
print("Number of connected components:", num_components)

# components = list(nx.connected_components(G))

# small_components = [c for c in components if len(c) < 100]

# # Remove nodes in small components from the graph
# for component in small_components:
#     G.remove_nodes_from(component)

# # print("Removed all components with fewer than 30 nodes.")

# num_components = nx.number_connected_components(G)
# print("Number of connected components:", num_components)

# nx.write_graphml(G, "ppi_network.graphml")


# nodes_data = pd.DataFrame(G.nodes(data=True))
# nodes_data.columns = ['Node', 'Attributes']
# attributes_df = nodes_data['Attributes'].apply(pd.Series)
# result_df = pd.concat([nodes_data['Node'], attributes_df], axis=1)
# result_df.to_csv("nodes.csv", index=False)

# edges_data = pd.DataFrame(G.edges(data=True))
# edges_data.columns = ['Source', 'Target', 'Attributes']
# edges_data = edges_data.drop(columns=["Attributes"])
# edges_data.to_csv("edges.csv", index=False)

components = list(nx.connected_components(G))

for i, component_nodes in enumerate(components):
    # Create a subgraph for the component
    component_subgraph = G.subgraph(component_nodes).copy()
    
    # Save the subgraph to a file
    graphml_filename = f"data/graphs/graphml/component_{i}.graphml"
    nx.write_graphml(component_subgraph, graphml_filename)
    edgelist_filename = f"data/graphs/edge_lists/component_{i}.edgelist"
    nx.write_edgelist(component_subgraph, edgelist_filename, data=False, delimiter="\t")

