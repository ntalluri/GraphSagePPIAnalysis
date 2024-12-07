import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import os
from glob import glob
from sklearn.metrics import jaccard_score

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
    Calculate Jaccard similarity between two graphs using sklearn.
    
    Args:
        graph1 (nx.Graph): First graph to compare
        graph2 (nx.Graph): Second graph to compare
        
    Returns:
        tuple: (node_similarity, edge_similarity) as Jaccard indices
    """
    # for nodes: Convert to binary indicators
    all_nodes = list(set(graph1.nodes()).union(graph2.nodes()))
    nodes1 = [1 if node in graph1.nodes() else 0 for node in all_nodes]
    nodes2 = [1 if node in graph2.nodes() else 0 for node in all_nodes]
    node_similarity = jaccard_score(nodes1, nodes2)
    
    # for edges: Convert to binary indicators
    all_edges = list(set(graph1.edges()).union(graph2.edges()))
    edges1 = [1 if edge in graph1.edges() else 0 for edge in all_edges]
    edges2 = [1 if edge in graph2.edges() else 0 for edge in all_edges]
    edge_similarity = jaccard_score(edges1, edges2)
    
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



def save_similarity_distributions(node_sims, edge_sims, graph1_files, graph2_files, title, filename):
    """
    Plot and save histograms of node and edge similarities between graph sets.
    
    Creates two side-by-side plots:
    1. Node similarities histogram with KDE curve
    2. Edge similarities histogram with KDE curve
    
    Interpretation:
    - X-axis: Jaccard similarity (0-1)
    - Y-axis: Frequency of similarity scores
    - Higher peaks in distribution indicate common similarity values
    - Right-skewed distribution suggests high similarity between graphs
    - Left-skewed distribution suggests low similarity between graphs

    Args:
        node_sims (array): Array of node similarity scores
        edge_sims (array): Array of edge similarity scores
        graph1_files (list): List of first graph filenames
        graph2_files (list): List of second graph filenames
        title (str): Plot title
        filename (str): Output filename prefix
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

def plot_single_graph_comparison(target_graph_file, target_graph, train_graphs, output_dir, prefix):
    """
    Create histogram comparing one graph against all training graphs.
    
    Args:
        target_graph_file (str): Filename of the target graph (test/val)
        target_graph (nx.Graph): Target graph object
        train_graphs (list): List of (filename, graph) tuples for training graphs
        output_dir (str): Output directory
        prefix (str): Prefix for output filename ('test' or 'val')
    """
    node_sims = []
    edge_sims = []
    train_files = []
    
    for train_file, train_graph in train_graphs:
        node_sim, edge_sim = calculate_jaccard_similarity(target_graph, train_graph)
        node_sims.append(node_sim)
        edge_sims.append(edge_sim)
        train_files.append(os.path.basename(train_file))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    target_name = os.path.basename(target_graph_file)
    
    def plot_distribution(data, ax, title_suffix):
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        sns.histplot(data, bins=len(train_files), ax=ax)
        kde = sns.kdeplot(data, ax=ax, color='red', linewidth=2)
        
        kde_line = kde.lines[0]
        x_data, y_data = kde_line.get_data()
        
        mean_idx = np.abs(x_data - mean_val).argmin()
        median_idx = np.abs(x_data - median_val).argmin()
        
        ax.plot(mean_val, y_data[mean_idx], 'go', markersize=8, label=f'Mean: {mean_val:.3f}')
        ax.plot(median_val, y_data[median_idx], 'bo', markersize=8, label=f'Median: {median_val:.3f}')
        
        ax.set_title(f'{title_suffix}\nComparison: {target_name} vs Training Graphs')
        ax.set_xlabel('Jaccard Similarity Score (0-1)')
        ax.set_ylabel('Frequency (Number of Training Graphs)')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f'{x:.2f}' for x in np.linspace(0, 1, 11)], rotation=45, ha='right')
        ax.legend()
    
    plot_distribution(node_sims, ax1, 'Node Similarities')
    plot_distribution(edge_sims, ax2, 'Edge Similarities')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}_single_{target_name}_dist.png', bbox_inches='tight')
    plt.close()
    

def save_similarity_heatmap(similarities, graph1_files, graph2_files, title, filename):
    """
    Create and save a heatmap visualization of graph similarities.
    """
    plt.figure(figsize=(12, 8))
    
    # Create a matrix using unique filenames
    unique_graph1 = list(dict.fromkeys(graph1_files))
    unique_graph2 = list(dict.fromkeys(graph2_files))
    sim_matrix = np.zeros((len(unique_graph1), len(unique_graph2)))
    
    # Fill the matrix with similarities
    for idx, (g1, g2, sim) in enumerate(zip(graph1_files, graph2_files, similarities)):
        i = unique_graph1.index(g1)
        j = unique_graph2.index(g2)
        sim_matrix[i, j] = sim
    
    # Create heatmap
    sns.heatmap(sim_matrix, fmt=".2f", cmap="coolwarm", 
                cbar=True, 
                xticklabels=[os.path.basename(f) for f in unique_graph2],
                yticklabels=[os.path.basename(f) for f in unique_graph1])
    
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
    df.to_csv(f'{filename}_similarities.csv', index=False, sep="\t")

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
    


    # histograms
    save_similarity_distributions(train_train_nodes, train_train_edges, 
                                train_graph_one, train_graph_two,
                                "Train-Train Comparison", f"{output_dir}/train_train")
    save_similarity_distributions(test_train_nodes, test_train_edges, 
                                test_graph_one, train_graph_three,
                                "Test-Train Comparison", f"{output_dir}/test_train")
    save_similarity_distributions(val_train_nodes, val_train_edges, 
                                val_graph_one, train_graph_four,
                                "Val-Train Comparison", f"{output_dir}/val_train")
    
    # compare each test graph
    for test_file, test_graph in load_graphs(test_dir):
        plot_single_graph_comparison(test_file, test_graph, train_graphs, output_dir, 'test')
        # df.to_csv(f'{output_dir}/test_single_{os.path.basename(test_file)}_similarities.csv', index=False)
    
    # compare each validation graph
    for val_file, val_graph in load_graphs(val_dir):
        plot_single_graph_comparison(val_file, val_graph, train_graphs, output_dir, 'val')
        # df.to_csv(f'{output_dir}/val_single_{os.path.basename(val_file)}_similarities.csv', index=False)

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

    # save jacard index to a file
    save_similarities(train_graph_one, train_graph_two, train_train_nodes, train_train_edges, f"{output_dir}/train_train")
    save_similarities(test_graph_one, train_graph_three, test_train_nodes, test_train_edges, f"{output_dir}/test_train")
    save_similarities(val_graph_one, train_graph_four, val_train_nodes, val_train_edges, f"{output_dir}/val_train")

# run code
train_directory = "data/matching-graphs/train"
val_directory = "data/matching-graphs/val"
test_directory = "data/matching-graphs/test"
analyze_graph_similarities(train_dir=train_directory, val_dir=val_directory, test_dir=test_directory)

