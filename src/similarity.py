import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import os
from glob import glob

def load_graphs(directory, pattern="*.graphml"):
    """
    Load all GraphML files from a directory into NetworkX graph objects.
    
    Args:
        directory (str): Path to directory containing GraphML files
        pattern (str): File pattern to match GraphML files
        
    Returns:
        list: Tuples of (filename, NetworkX graph) pairs
    """

    graph_files = glob(os.path.join(directory, pattern))
    return [(f, nx.read_graphml(f)) for f in graph_files]

def calculate_jaccard_similarity(graph1, graph2):
    """
    Calculate Jaccard similarity between two graphs for both nodes and edges.
    
    Args:
        graph1 (nx.Graph): First graph to compare
        graph2 (nx.Graph): Second graph to compare
        
    Returns:
        tuple: (node_similarity, edge_similarity) as Jaccard indices
    """

    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    node_intersection = len(nodes1.intersection(nodes2))
    node_union = len(nodes1.union(nodes2))
    node_similarity = node_intersection / node_union if node_union > 0 else 0

    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    edge_intersection = len(edges1.intersection(edges2))
    edge_union = len(edges1.union(edges2))
    edge_similarity = edge_intersection / edge_union if edge_union > 0 else 0

    return node_similarity, edge_similarity

def compute_pairwise_similarities(graphs1, graphs2):
    """
    Compute pairwise Jaccard similarities between two sets of graphs.
    
    Args:
        graphs1 (list): First list of (filename, graph) tuples
        graphs2 (list): Second list of (filename, graph) tuples
        
    Returns:
        tuple: (graph1_files, graph2_files, node_similarities, edge_similarities)
    """

    graph_one_files = []
    graph_two_files = []
    node_similarities = []
    edge_similarities = []
    
    for (f1, g1), (f2, g2) in product(graphs1, graphs2):
        node_sim, edge_sim = calculate_jaccard_similarity(g1, g2)
        graph_one_files.append(os.path.basename(f1))
        graph_two_files.append(os.path.basename(f2))
        node_similarities.append(node_sim)
        edge_similarities.append(edge_sim)
    
    return graph_one_files, graph_two_files, np.array(node_similarities), np.array(edge_similarities)


# def save_similarity_distributions(node_sims, edge_sims, graph1_files, graph2_files, title, filename):
#     """
#     Plot and save histograms of node and edge similarities between graph sets.
    
#     Creates two side-by-side plots:
#     1. Node similarities histogram with KDE curve
#     2. Edge similarities histogram with KDE curve
    
#     Interpretation:
#     - X-axis: Jaccard similarity (0-1)
#     - Y-axis: Frequency of similarity scores
#     - Higher peaks in distribution indicate common similarity values
#     - Right-skewed distribution suggests high similarity between graphs
#     - Left-skewed distribution suggests low similarity between graphs

#     Args:
#         node_sims (array): Array of node similarity scores
#         edge_sims (array): Array of edge similarity scores
#         graph1_files (list): List of first graph filenames
#         graph2_files (list): List of second graph filenames
#         title (str): Plot title
#         filename (str): Output filename prefix
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
#     graph_labels = [f"{g1}\n{g2}" for g1, g2 in zip(graph1_files, graph2_files)]
#     n_bins = len(set(graph_labels))
#     print(n_bins)
#     # n_bins = 20

#     sns.histplot(node_sims, bins=n_bins, ax=ax1)
#     sns.kdeplot(node_sims, ax=ax1, color='red', linewidth=2)
#     ax1.set_title(f'Node Similarities\n{title}')
#     ax1.set_xlabel('Jaccard Similarity')
#     ax1.set_xlim(0, 1)
#     ticks = np.linspace(0, 1, 11)
#     ax1.set_xticks(ticks)
#     ax1.set_xticklabels([f'{x:.2f}' for x in ticks], rotation=45, ha='right')

    
#     sns.histplot(edge_sims, bins=n_bins, ax=ax2)
#     sns.kdeplot(edge_sims, ax=ax2, color='red', linewidth=2)
#     ax2.set_title(f'Edge Similarities\n{title}')
#     ax2.set_xlabel('Jaccard Similarity')
#     ax2.set_xlim(0, 1)  # Set x-axis limits
#     ticks = np.linspace(0, 1, 11)
#     ax2.set_xticks(ticks)
#     ax2.set_xticklabels([f'{x:.2f}' for x in ticks], rotation=45, ha='right')

#     plt.tight_layout()
#     plt.savefig(f'{filename}_dist.png', bbox_inches='tight')
#     plt.close()

def save_similarity_distributions(node_sims, edge_sims, graph1_files, graph2_files, title, filename):
    """
    Plot and save histograms of node and edge similarities with statistical measures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    graph_labels = [f"{g1}\n{g2}" for g1, g2 in zip(graph1_files, graph2_files)]
    n_bins = len(set(graph_labels))
    # n_bins = 20
    
    def plot_distributions(data, ax, title_suffix):
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        sns.histplot(data, bins=n_bins, ax=ax)
        kde = sns.kdeplot(data, ax=ax, color='red', linewidth=2)
        
        # Add dots on the KDE line for statistics
        kde_line = kde.lines[0]
        x_data, y_data = kde_line.get_data()
        
        # Find y-values on KDE line corresponding to statistics
        mean_idx = np.abs(x_data - mean_val).argmin()
        median_idx = np.abs(x_data - median_val).argmin()

        ax.plot(mean_val, y_data[mean_idx], 'go', markersize=8, label=f'Mean: {mean_val:.3f}')
        ax.plot(median_val, y_data[median_idx], 'bo', markersize=8, label=f'Median: {median_val:.3f}')
        
        ax.set_title(f'{title_suffix}\n{title}')
        ax.set_xlabel('Jaccard Similarity')
        ax.set_xlim(0, 1)
        ticks = np.linspace(0, 1, 11)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x:.2f}' for x in ticks], rotation=45, ha='right')
        ax.legend()
    
    plot_distributions(node_sims, ax1, 'Node Similarities')
    plot_distributions(edge_sims, ax2, 'Edge Similarities')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_dist.png', bbox_inches='tight')
    plt.close()

def save_similarity_heatmap(similarities, graph1_files, graph2_files, title, filename):
    """
    Create and save a heatmap visualization of graph similarities.
    
    Creates a heatmap where:
    - X-axis: Second set of graph files
    - Y-axis: First set of graph files
    - Color intensity: Similarity score (0-1)
        - Darker red indicates higher similarity (closer to 1)
        - Darker blue indicates lower similarity (closer to 0)
    
    Interpretation:
    - Diagonal patterns indicate consistent similarities between specific graphs
    - Uniform colors suggest consistent similarity across all comparisons
    - Scattered hot/cold spots indicate varying similarities between graph pairs
    

    Args:
        similarities (array): Matrix of similarity scores
        graph1_files (list): List of first graph filenames
        graph2_files (list): List of second graph filenames
        title (str): Plot title
        filename (str): Output filename prefix
    """

    plt.figure(figsize=(12, 8))
    
    # Reshape similarities into a matrix
    n_graphs1 = len(set(graph1_files))
    n_graphs2 = len(set(graph2_files))
    sim_matrix = similarities.reshape(n_graphs1, n_graphs2)
    
    # Create heatmap with graph labels
    sns.heatmap(sim_matrix, fmt=".2f", cmap="coolwarm", 
                cbar=True, 
                xticklabels=[os.path.basename(f) for f in set(graph2_files)],
                yticklabels=[os.path.basename(f) for f in set(graph1_files)])
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{filename}_heatmap.png', bbox_inches='tight')
    plt.close()

def save_similarities(graph1_files, graph2_files, node_sims, edge_sims, filename):
    """
    Save graph similarity scores to a CSV file.
    
    Args:
        graph1_files (list): List of first graph filenames
        graph2_files (list): List of second graph filenames
        node_sims (array): Array of node similarity scores
        edge_sims (array): Array of edge similarity scores
        filename (str): Output filename prefix
    """
    df = pd.DataFrame({
        'graph1': graph1_files,
        'graph2': graph2_files,
        'node_similarities': node_sims,
        'edge_similarities': edge_sims
    })
    df.to_csv(f'{filename}_similarities.csv', index=False)

def analyze_graph_similarities(train_dir, val_dir, test_dir, output_dir='output'):
    """
    Analyze and visualize similarities between graphs in training, testing, and validation sets.
    
    Args:
        train_dir (str): Directory containing training graphs
        test_dir (str): Directory containing testing graphs
        val_dir (str): Directory containing validation graphs
        output_dir (str): Directory for saving output files
    """
    train_graphs = load_graphs(train_dir)
    val_graphs = load_graphs(val_dir)
    test_graphs = load_graphs(test_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_graph_one, train_graph_two, train_train_nodes, train_train_edges = compute_pairwise_similarities(train_graphs, train_graphs)
    test_graph_one, train_graph_three, test_train_nodes, test_train_edges = compute_pairwise_similarities(test_graphs, train_graphs)
    val_graph_one, train_graph_four, val_train_nodes, val_train_edges = compute_pairwise_similarities(val_graphs, train_graphs)
    


    save_similarity_distributions(train_train_nodes, train_train_edges, 
                                train_graph_one, train_graph_two,
                                "Train-Train Comparison", f"{output_dir}/train_train")
    save_similarity_distributions(test_train_nodes, test_train_edges, 
                                test_graph_one, train_graph_three,
                                "Test-Train Comparison", f"{output_dir}/test_train")
    save_similarity_distributions(val_train_nodes, val_train_edges, 
                                val_graph_one, train_graph_four,
                                "Val-Train Comparison", f"{output_dir}/val_train")
    

    # Save heatmaps
    save_similarity_heatmap(train_train_nodes, train_graph_one, train_graph_two,
                           "Train-Train Node Similarities", f"{output_dir}/train_train_nodes")
    save_similarity_heatmap(train_train_edges, train_graph_one, train_graph_two,
                           "Train-Train Edge Similarities", f"{output_dir}/train_train_edges")

    save_similarity_heatmap(test_train_nodes, test_graph_one, train_graph_three,
                           "Test-Train Node Similarities", f"{output_dir}/test_train_nodes")
    save_similarity_heatmap(test_train_edges, test_graph_one, train_graph_three,
                           "Test-Train Edge Similarities", f"{output_dir}/test_train_edges")

    save_similarity_heatmap(val_train_nodes, val_graph_one, train_graph_four,
                           "Val-Train Node Similarities", f"{output_dir}/val_train_nodes")
    save_similarity_heatmap(val_train_edges, val_graph_one, train_graph_four,
                           "Val-Train Edge Similarities", f"{output_dir}/val_train_edges")

    
    save_similarities(train_graph_one, train_graph_two, train_train_nodes, train_train_edges, f"{output_dir}/train_train")
    save_similarities(test_graph_one, train_graph_three, test_train_nodes, test_train_edges, f"{output_dir}/test_train")
    save_similarities(val_graph_one, train_graph_four, val_train_nodes, val_train_edges, f"{output_dir}/val_train")

# run code
train_directory = "data/matching-graphs/train"
val_directory = "data/matching-graphs/val"
test_directory = "data/matching-graphs/test"
analyze_graph_similarities(train_dir=train_directory, val_dir=val_directory, test_dir=test_directory)