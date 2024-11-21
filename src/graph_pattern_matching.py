# Will give the OHMNET graphs that match with the components with better organization than what is above
# uses my preset list based on the pattern I found to find those graphs

import networkx as nx
from pathlib import Path

def load_graph(file_path):
    return nx.read_edgelist(file_path, delimiter="\t")

def save_graph(graph, output_path):
    nx.write_graphml(graph, output_path)

def create_edge_count_index(folder):
    """
    Load all graphs from the given folder and index them by their edge counts.
    Returns a dictionary mapping edge_count to list of (filename, graph).
    """
    folder = Path(folder)
    edge_count_dict = {}
    for graph_file in folder.glob("*.edgelist"):
        graph = load_graph(graph_file)
        edge_count = len(graph.edges())
        if edge_count not in edge_count_dict:
            edge_count_dict[edge_count] = []
        edge_count_dict[edge_count].append((graph_file.name, graph))
    return edge_count_dict

def find_matching_graphs_with_mapping(folder1, folder2, output_folder, edge_mapping):
    """
    Find and save matching graphs based on a predefined edge count mapping.

    Parameters:
    - folder1: Path to biotissue graphs.
    - folder2: Path to component graphs.
    - output_folder: Path to save the organized graphs.
    - edge_mapping: Dictionary mapping graph2 edge counts to graph1 edge counts(s).
    """
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Index graphs by their edge counts
    print("Indexing folder1 (biotissue graphs)...")
    folder1_index = create_edge_count_index(folder1)
    print(f"Indexed {sum(len(v) for v in folder1_index.values())} biotissue graphs.")

    print("Indexing folder2 (component graphs)...")
    folder2_index = create_edge_count_index(folder2)
    print(f"Indexed {sum(len(v) for v in folder2_index.values())} component graphs.")

    match_dict = {}

    for graph2_edge, graph1_edges in edge_mapping.items():
        graph2_entries = folder2_index.get(graph2_edge, [])
        if not graph2_entries:
            print(f"No component graph found with {graph2_edge} edges.")
            continue  # Skip if no matching graph2

        # graph1_edges can be a single integer or a list
        if isinstance(graph1_edges, int):
            graph1_edges = [graph1_edges]
        elif isinstance(graph1_edges, str):
            graph1_edges = [int(e.strip()) for e in graph1_edges.split(",")]

        for graph2_name, graph2 in graph2_entries:
            matching_graph1s = []
            for edge_count1 in graph1_edges:
                graph1_entries = folder1_index.get(edge_count1, [])
                if not graph1_entries:
                    print(f"No biotissue graph found with {edge_count1} edges for component graph {graph2_name}.")
                    continue
                matching_graph1s.extend(graph1_entries)

            if not matching_graph1s:
                print(f"No matching biotissue graphs found for component graph {graph2_name}.")
                continue

            # Create a subfolder for this component graph
            component_subfolder = output_folder / f"{graph2_name}"
            component_subfolder.mkdir(parents=True, exist_ok=True)

            # Save the component graph with the desired filename
            graph2_filename = Path(graph2_name).stem + ".graphml"  # Remove '.edgelist' and add '.graphml'
            graph2_output_path = component_subfolder / graph2_filename
            save_graph(graph2, graph2_output_path)

            # Save each matching biotissue graph with the desired filename
            for graph1_name, graph1 in matching_graph1s:
                graph1_stem = Path(graph1_name).stem  # Remove '.edgelist'
                graph1_filename = f"{graph1_stem}_{len(graph1.edges())}.graphml"
                graph1_output_path = component_subfolder / graph1_filename
                save_graph(graph1, graph1_output_path)
                match_dict[graph2_name] = match_dict.get(graph2_name, []) + [graph1_filename]

    return match_dict

# Define the edge count mapping 24 components edge counts to Ohmnet edge counts
edge_mapping = {
    31655: 31665,
    51648: 51666,
    52108: 52126,
    50047: 50061,
    46940: 46956,
    48394: 48409,
    43950: 43964,
    39657: 39671,
    30986: 30993,
    43841: 43854,
    54806: 54824,
    23735: 23741,
    24010: 24016,
    33195: [33201, 33281],
    54539: 54554,
    4182: 4193,
    36264: 36276,
    23029: 23041,
    9627: [9629, 9635],
    18862: 18871,
    33721: 33731,
    30829: 30839,
    15506: [15522, 15580, 15517],
    16904: 16912,
}

folder1_path = "data/bio-tissue-networks" 
folder2_path = "data/graphs/edge_lists"      
output_folder_path = "output/" 

matches = find_matching_graphs_with_mapping(
    folder1_path, 
    folder2_path, 
    output_folder_path, 
    edge_mapping
)

with open(output_folder_path + "graph_matches.txt", "w") as f:
    for graph2, matching_graph1s in matches.items():
        match_str = ", ".join(matching_graph1s) if matching_graph1s else "No matches"
        f.write(f"{graph2} matches with: {match_str}\n")

print("Matching graphs saved to .graphml format in the output folder.")
print("Match summary saved to graph_matches.txt.")
######################################################################################################################################################

# Will give the OHMNET graphs that match with the components based on finding the min edges that are greater than the component edges

# import networkx as nx
# from pathlib import Path

# def load_graph(file_path):
#     """Load a graph from an edgelist file."""
#     return nx.read_edgelist(file_path, delimiter="\t")

# def save_graph(graph, output_path):
#     """Save a graph to a .graphml file."""
#     nx.write_graphml(graph, output_path)

# def find_matching_graphs(folder1, folder2, output_folder):
#     """Find matching graphs between two folders where graph2 edge count < graph1 edge count."""
#     folder1 = Path(folder1)
#     folder2 = Path(folder2)
#     output_folder = Path(output_folder)
#     output_folder.mkdir(parents=True, exist_ok=True)

#     match_dict = {}
#     graph2_dict = {}

#     # Load graphs from folder2
#     print("Loading graph2 files...")
#     for graph_file2 in folder2.glob("*.edgelist"):
#         graph2 = load_graph(graph_file2)
#         graph2_dict[graph_file2.name] = (graph2, len(graph2.edges()))
#     print(f"Loaded {len(graph2_dict)} graph2 files.")

#     # Compare graphs from folder1 to folder2
#     for graph_file1 in folder1.glob("*.edgelist"):
#         graph1 = load_graph(graph_file1)
#         edge_count1 = len(graph1.edges())
#         print(f"Processing {graph_file1.name} with {edge_count1} edges.")
#         closest_matches = []
#         min_difference = float('inf')

#         # Find the closest match in edge count from folder2 where edge_count2 < edge_count1
#         for graph2_name, (graph2, edge_count2) in graph2_dict.items():
#             if edge_count2 < edge_count1:
#                 difference = edge_count1 - edge_count2
#                 if difference < min_difference:
#                     closest_matches = [graph2_name]
#                     min_difference = difference
#                 elif difference == min_difference:
#                     closest_matches.append(graph2_name)

#         # Save the graph and its matches
#         graph1_output_folder = output_folder / graph_file1.stem
#         graph1_output_folder.mkdir(parents=True, exist_ok=True)
#         save_graph(graph1, graph1_output_folder / f"{graph_file1.stem}.graphml")

#         for match_name in closest_matches:
#             graph2, _ = graph2_dict[match_name]
#             save_graph(graph2, graph1_output_folder / f"{match_name}.graphml")

#         match_dict[graph_file1.name] = closest_matches

#     return match_dict

# folder1_path = "data/bio-tissue-networks"
# folder2_path = "data/graphs/edge_lists"
# output_folder_path = "output/matching_graphs"

# matches = find_matching_graphs(folder1_path, folder2_path, output_folder_path)

# # Save matches summary to a file
# with open("graph_matches.txt", "w") as f:
#     for graph1, matching_graphs in matches.items():
#         match_str = ", ".join(matching_graphs) if matching_graphs else "No matches"
#         f.write(f"{graph1} matches with: {match_str}\n")

# print("Matching graphs saved to .graphml format in the output folder.")
