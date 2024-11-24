import networkx as nx
import json
import pandas as pd
import os
import numpy as np

os.makedirs("data/graphs/csvs", exist_ok=True)
os.makedirs("data/graphs/edge_lists", exist_ok=True)

with open('data/ppi/ppi-G.json', 'r') as file:
    ppi_graph = json.load(file)

ppi_feats = np.load('data/ppi/ppi-feats.npy') # assuming in the right order based on invesitgation done on toy graph

G = nx.Graph()
for node in ppi_graph['nodes']:
    node_id = node.get('id')
    G.add_node(node_id, **node, features=list(ppi_feats[node_id]))

for link in ppi_graph['links']:
    source = link.get('source')
    target = link.get('target')
    G.add_edge(source, target)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is the graph connected?", nx.is_connected(G))
print("Number of connected components:", nx.number_connected_components(G))

# nx.write_graphml(G, "data/ppi_network.graphml")

components = list(nx.connected_components(G))
for i, component_nodes in enumerate(components):
    # create a subgraph for the component
    component_subgraph = G.subgraph(component_nodes).copy()
    
    if component_subgraph.number_of_edges() > 5:

        # determine graph type
        all_test = all(data.get('test', False) for _, data in component_subgraph.nodes(data=True))
        all_val = all(data.get('val', False) for _, data in component_subgraph.nodes(data=True))
        if all_test:
            graph_type = 'test'
        elif all_val:
            graph_type = 'validation'
        else:
            graph_type = 'train'
        
        # save the edgelists
        edgelist_filename = f"data/graphs/edge_lists/component_{i}_graphtype_{graph_type}_numedges{component_subgraph.number_of_edges()}.edgelist"
        nx.write_edgelist(component_subgraph, edgelist_filename, data=False, delimiter="\t")

        # save node and edge attributes in csvs to put in cytoscape
        nodes_data = []
        for node, attrs in G.nodes(data=True):
            node_dict = {}
            node_dict.update(attrs)
            nodes_data.append(node_dict)

        nodes_df = pd.DataFrame(nodes_data)
        nodes_csv_filename = f"data/graphs/csvs/nodes_component_{i}_graphtype_{graph_type}_numedges_{component_subgraph.number_of_edges()}.csv"
        nodes_df.to_csv(nodes_csv_filename, index=False, sep="\t")

        edges_data = []
        for source, target, attrs in G.edges(data=True):
            edge_dict = {'source': source, 'target': target}
            edge_dict.update(attrs)
            edges_data.append(edge_dict)

        edges_df = pd.DataFrame(edges_data)
        edges_csv_filename = f"data/graphs/csvs/edges_component_{i}_graphtype_{graph_type}_numedges_{component_subgraph.number_of_edges()}.csv"
        edges_df.to_csv(edges_csv_filename, index=False, sep="\t")
