from dgl.data import PPIDataset
import dgl
import networkx as nx
from torch_geometric.datasets import PPI

train_dataset = PPI(root="data", split="train")
val_dataset = PPI(root="data", split="val")
test_dataset = PPI(root="data", split="test")
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))
print(type(test_dataset[0]))

for graph in train_dataset:
    print(f"Train graph nodes: {graph.num_nodes}")
    # print(f"Train graph nodes: {graph.num_edges}")
for graph in val_dataset:
    print(f"Val graph nodes: {graph.num_nodes}")
    # print(f"Val graph nodes: {graph.num_edges}")
for graph in test_dataset:
    print(f"Test graph nodes: {graph.num_nodes}")
    # print(f"Test graph nodes: {graph.num_edges}")

# seeing if I can use the DGL graphs
train_dataset = dgl.data.PPIDataset(mode='train')
val_dataset = PPIDataset(mode='valid')
test_dataset = PPIDataset(mode='test')

print("test")
for g in test_dataset:
    print("num of nodes:", g.number_of_nodes())
    print("num of edges:", g.number_of_edges())

print("true test vals")
graph1 = nx.read_edgelist("data/bio-tissue-networks/lung.edgelist")
print("num of nodes:", graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

graph1 = nx.read_edgelist("data/bio-tissue-networks/midbrain.edgelist")
print("num of nodes:", graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

print("val")
for g in val_dataset:
    print("num of nodes:", g.number_of_nodes())
    print("num of edges:", g.number_of_edges())

print("true val vals")
graph1 = nx.read_edgelist("data/bio-tissue-networks/heart.edgelist")
print("num of nodes:", graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

graph1 = nx.read_edgelist("data/bio-tissue-networks/kidney.edgelist")
print("num of nodes:",graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

# Note the nodes are placed in the wrong graphs for the DGL graphs

