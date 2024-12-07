import dgl
from dgl.data import PPIDataset
import networkx as nx

# seeing if I can use the DGL graphs
train_dataset = PPIDataset(mode='train')
val_dataset = PPIDataset(mode='valid')
test_dataset = PPIDataset(mode='test')

print("test")
for g in test_dataset:
    print("num of nodes:", g.number_of_nodes())
    print("num of edges:", g.number_of_edges())


print("true vals")
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

print("true vals")
graph1 = nx.read_edgelist("data/bio-tissue-networks/heart.edgelist")
print("num of nodes:", graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

graph1 = nx.read_edgelist("data/bio-tissue-networks/kidney.edgelist")
print("num of nodes:",graph1.number_of_nodes())
print("num of edges:", graph1.number_of_edges())

# Note the nodes are placed in the wrong graphs for the DGL graphs

