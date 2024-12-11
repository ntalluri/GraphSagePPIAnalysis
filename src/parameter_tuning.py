# Parameter Tuning
import torch
import numpy as np
from itertools import product
from sklearn.metrics import f1_score
import time
from torch_geometric.datasets import PPI
from dgl.data import PPIDataset
import random 
import dgl
import networkx as nx
from itertools import permutations
from sklearn.metrics import jaccard_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch.optim import Adam
import torch.nn.functional as F

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
    
    # Split into new train/val/test
    val_graphs = reordered_graphs[:2]
    test_graphs = reordered_graphs[2:4]
    train_graphs = reordered_graphs[4:]
    
    return {
        'train': train_graphs,
        'val': val_graphs,
        'test': test_graphs
    }

# new split dataset
# Searching for a split
# print("searching")
# while True:
#     result = distribute_graphs()
#     if result['test_score'] <= 0.4 and result['val_score'] <= 0.7:
#         print(f"Found distribution with score: {result['test_score']:.3f}, {result['val_score']:.3f}")
#         print(result['selected_indices'])
#         break

# create new splits using your indices
# splits = reorder_graphs(result['selected_indices'])


def grid_search_graphsage():
    # define parameter grid
    param_grid = {
        'hidden_channels': [64, 128, 256, 512],
        'num_layers': [1, 2, 3, 4],
        'dropout': [0.1, 0.3, 0.5, 0.7],
        'aggr': ['mean', 'max'],
        'normalize': [True, False],
        'batch': [1,2]
    }

     # generate all combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    best_score = 0
    best_params = None
    results = []

    
    for params in param_combinations:
        start_time = time.time()
        print(f"Params: {params}")
        
        # # noraml dataset
        # splits = reorder_graphs([5, 12, 1, 8, 4, 20, 3, 14, 10, 18, 22, 11, 7, 2, 21, 13, 0, 6, 15, 9, 19, 17, 23, 16])
        # train_loader = DataLoader(splits['train'], batch_size=params['batch'], shuffle=True)
        # val_loader = DataLoader(splits['val'], batch_size=params['batch'], shuffle=False)
        # test_loader = DataLoader(splits['test'], batch_size=params['batch'], shuffle=False)

        # norm split
        train_dataset = PPI(root="data", split="train")
        val_dataset = PPI(root="data", split="val")
        test_dataset = PPI(root="data", split="test")
        train_loader = DataLoader(train_dataset, batch_size=params['batch'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params['batch'], shuffle=False)
       
        # Initialize model with current parameters
        model = GraphSAGE(
            in_channels=50,
            hidden_channels=params['hidden_channels'],
            num_layers=params['num_layers'],
            out_channels=121,
            dropout=params['dropout'],
            aggr=params['aggr'],
            normalize=params['normalize']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # train for a few epochs
        model.train()
        for epoch in range(5):
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y)
                loss.backward()
                optimizer.step()
        
        # evaluate
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data in val_loader:
                out = model(data.x, data.edge_index)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float().cpu().numpy()
                true = data.y.cpu().numpy()
                y_pred.append(preds)
                y_true.append(true)
        
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        
        duration = time.time() - start_time
        
        results.append({
            'params': params,
            'f1_micro': f1_micro,
            'duration': duration
        })
        
        if f1_micro > best_score:
            best_score = f1_micro
            best_params = params
            
        print(f"F1 Score: {f1_micro:.4f}")
        print(f"Duration: {duration:.2f}s\n")
    
    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print(f"Best F1 Score: {best_score:.4f}")
    
    return best_params, results

# run grid search
best_params, results = grid_search_graphsage()
