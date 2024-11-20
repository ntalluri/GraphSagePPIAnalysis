import networkx as nx
from pathlib import Path

def load_graph(file_path):
    """Load a graph from an edgelist file."""
    return nx.read_edgelist(file_path, delimiter="\t")

def save_graph(graph, output_path):
    """Save a graph to a .graphml file."""
    nx.write_graphml(graph, output_path)

def find_matching_graphs(folder1, folder2, output_folder):
    """Find matching graphs between two folders where graph2 edge count < graph1 edge count."""
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    match_dict = {}
    graph2_dict = {}

    # Load graphs from folder2
    print("Loading graph2 files...")
    for graph_file2 in folder2.glob("*.edgelist"):
        graph2 = load_graph(graph_file2)
        graph2_dict[graph_file2.name] = (graph2, len(graph2.edges()))
    print(f"Loaded {len(graph2_dict)} graph2 files.")

    # Compare graphs from folder1 to folder2
    for graph_file1 in folder1.glob("*.edgelist"):
        graph1 = load_graph(graph_file1)
        edge_count1 = len(graph1.edges())
        print(f"Processing {graph_file1.name} with {edge_count1} edges.")
        closest_matches = []
        min_difference = float('inf')

        # Find the closest match in edge count from folder2 where edge_count2 < edge_count1
        for graph2_name, (graph2, edge_count2) in graph2_dict.items():
            if edge_count2 < edge_count1:
                difference = edge_count1 - edge_count2
                if difference < min_difference:
                    closest_matches = [graph2_name]
                    min_difference = difference
                elif difference == min_difference:
                    closest_matches.append(graph2_name)

        # Save the graph and its matches
        graph1_output_folder = output_folder / graph_file1.stem
        graph1_output_folder.mkdir(parents=True, exist_ok=True)
        save_graph(graph1, graph1_output_folder / f"{graph_file1.stem}.graphml")

        for match_name in closest_matches:
            graph2, _ = graph2_dict[match_name]
            save_graph(graph2, graph1_output_folder / f"{match_name}.graphml")

        match_dict[graph_file1.name] = closest_matches

    return match_dict

folder1_path = "data/bio-tissue-networks"
folder2_path = "data/graphs/edge_lists"
output_folder_path = "output/matching_graphs"

matches = find_matching_graphs(folder1_path, folder2_path, output_folder_path)

# Save matches summary to a file
with open("graph_matches.txt", "w") as f:
    for graph1, matching_graphs in matches.items():
        match_str = ", ".join(matching_graphs) if matching_graphs else "No matches"
        f.write(f"{graph1} matches with: {match_str}\n")

print("Matching graphs saved to .graphml format in the output folder.")
