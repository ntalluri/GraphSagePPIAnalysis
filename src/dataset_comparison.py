from dgl.data import PPIDataset
import dgl
import networkx as nx
import torch
import numpy as np
from itertools import permutations
import random
from sklearn.metrics import jaccard_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score

# graph splits
def calculate_jaccard(graphs1, graphs2):
    total_score = 0
    comparisons = 0
    
    for g1 in graphs1:
        for g2 in graphs2:
            nodes1 = set(g1.nodes().tolist())
            nodes2 = set(g2.nodes().tolist())
            
            node_jaccard = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2))
            total_score += node_jaccard

    return total_score / (len(graphs1) * len(graphs2))


def distribute_graphs():
    train_dataset = list(PPIDataset(mode='train'))
    val_dataset = list(PPIDataset(mode='valid'))
    test_dataset = list(PPIDataset(mode='test'))

    all_graphs = (train_dataset + val_dataset + test_dataset)
    
    # randomly select graphs
    total_needed = 24  # 2 val + 2 test + 20 train
    selected_indices = random.sample(range(len(all_graphs)), total_needed)
    
    # distribute graphs
    val_indices = selected_indices[:2]
    test_indices = selected_indices[2:4]
    train_indices = selected_indices[4:]
    
    val_graphs = [all_graphs[i] for i in val_indices]
    test_graphs = [all_graphs[i] for i in test_indices]
    train_graphs = [all_graphs[i] for i in train_indices]
    
    # calculate jaccard score between test and val
    test_score = calculate_jaccard(test_graphs, train_graphs)
    val_score = calculate_jaccard(val_graphs, train_graphs)

    return {
        'selected_indices': selected_indices,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'train_graphs': train_graphs,
        'test_score': test_score,
        'val_score': val_score
    }


def reorder_graphs(indices):
    train_dataset = list(PPI(root="data", split="train"))
    val_dataset = list(PPI(root="data", split="val"))
    test_dataset = list(PPI(root="data", split="test"))
    all_graphs = train_dataset + val_dataset + test_dataset
    
    reordered_graphs = [all_graphs[i] for i in indices]
    
    # split into new train/val/test
    val_graphs = reordered_graphs[:2]
    test_graphs = reordered_graphs[2:4]
    train_graphs = reordered_graphs[4:]
    
    return {
        'train': train_graphs,
        'val': val_graphs,
        'test': test_graphs
    }

##################################################################################################################################################################################################################
print("New Split of Dataset")
# searching for a split
# print("searching")
# while True:
#     result = distribute_graphs()
#     if result['test_score'] <= 0.4 and result['val_score'] <= 0.7:
#         print(f"Found distribution with score: {result['test_score']:.3f}, {result['val_score']:.3f}")
#         print(result['selected_indices'])
#         break

# create new splits using new indices
# splits = reorder_graphs(result['selected_indices'])

# or use the best split I have found  with score: 0.398 test, 0.565 val
splits = reorder_graphs([5, 12, 1, 8, 4, 20, 3, 14, 10, 18, 22, 11, 7, 2, 21, 13, 0, 6, 15, 9, 19, 17, 23, 16])

# model and dataloader
# {'hidden_channels': 512, 'num_layers': 2, 'dropout': 0.1, 'aggr': 'mean', 'normalize': False, 'batch': 1}
in_channels = 50   # each node has 50 features
hidden_channels = 512
out_channels = 121  #  121 possible classes per node
num_layers = 2
dropout = 0.1
normalize = False
aggr = 'mean'

model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, aggr=aggr, normalize=normalize)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loader = DataLoader(splits['train'], batch_size=1, shuffle=True)
val_loader = DataLoader(splits['val'], batch_size=1, shuffle=False)
test_loader = DataLoader(splits['test'], batch_size=1, shuffle=False)

# model training
model.train()
for epoch in range(10):
    total_loss = 0
    for data in train_loader:
        # data is a single Graph object with x, edge_index, y
        # For PPI, y is multi-label (e.g. shape [num_nodes, 121])
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # for multi-label classification on PPI, BCEWithLogitsLoss is common.
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss {total_loss/len(train_loader)}")

# model eval with test and val
def evaluate(model, data_loader, split="val"):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            total_loss += loss.item()
            
            # compute predictions
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float().cpu().numpy()
            true = data.y.cpu().numpy()
            
            y_pred.append(preds)
            y_true.append(true)
    
    # calculate metrics
    avg_loss = total_loss / len(data_loader)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    
    print(f"{split} Loss: {avg_loss:.4f}")
    print(f"{split} F1 Score (micro): {f1_micro:.4f}")
    print(f"{split} F1 Score (macro): {f1_macro:.4f}")
    
    return avg_loss, f1_micro, f1_macro

val_loss, val_f1_micro, val_f1_macro = evaluate(model, val_loader, "val")
test_loss, test_f1_micro, test_f1_macro = evaluate(model, test_loader, "test")

####################################################################################################################################################################################
# normal dataset training 
print("Original Dataset")
train_dataset = PPI(root="data", split="train")
val_dataset = PPI(root="data", split="val")
test_dataset = PPI(root="data", split="test")

# create dataloaders and model
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Params: {'hidden_channels': 512, 'num_layers': 2, 'dropout': 0.1, 'aggr': 'mean', 'normalize': False, 'batch': 1}
in_channels = 50   # for PPI dataset, typically each node has 50 features.
hidden_channels = 512
out_channels = 121  # PPI dataset has 121 possible classes per node
num_layers = 2
dropout = 0.1
normalize = False
aggr = 'mean'

model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, aggr=aggr, normalize=normalize)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# model training
model.train()
for epoch in range(10):
    total_loss = 0
    for data in train_loader:
        # data is a single Graph object with x, edge_index, y
        # for PPI, y is multi-label (e.g. shape [num_nodes, 121])
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # for multi-label classification on PPI, BCEWithLogitsLoss is common.
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss {total_loss/len(train_loader)}")

# model eval for test and val
def evaluate(model, data_loader, split="val"):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            total_loss += loss.item()
            
            # Compute predictions
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float().cpu().numpy()
            true = data.y.cpu().numpy()
            
            y_pred.append(preds)
            y_true.append(true)
    
    # calculate metrics
    avg_loss = total_loss / len(data_loader)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print(f"{split} Loss: {avg_loss:.4f}")
    print(f"{split} F1 Score (micro): {f1_micro:.4f}")
    print(f"{split} F1 Score (macro): {f1_macro:.4f}")
    
    return avg_loss, f1_micro, f1_macro

val_loss, val_f1_micro, val_f1_macro = evaluate(model, val_loader, "val")
test_loss, test_f1_micro, test_f1_macro = evaluate(model, test_loader, "test")
