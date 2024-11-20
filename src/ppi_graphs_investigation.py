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
import os
import numpy as np

####################################################################################################################
with open('data/ppi/ppi-G.json', 'r') as file:
    ppi_graph = json.load(file)

# print(ppi_graph.keys())


# print(ppi_graph['directed']) # False
# print(ppi_graph['graph']) # empty?
# print(ppi_graph['multigraph']) # False
# print(ppi_graph['nodes']) # 56944 node id, test, val 
# print(ppi_graph['links']) # source and target

####################################################################################################################
ppi_feats = np.load('data/ppi/ppi-feats.npy')

# print(type(ppi_feats)) # nd array
# print(len(ppi_feats)) # 56944 features

# for feat in ppi_feats:
#     print(type(feat)) # nd array
#     print(len(feat)) # each of len 50
#     print(feat)

# We use positional gene sets, motif gene sets and immunological signatures as features  

# checking that 42% of nodes have no non-zero feature values
# ppi_feats = np.load('data/ppi/ppi-feats.npy')
# all_zero_rows = np.array([np.all(arr == 0) for arr in ppi_feats])
# num_all_zero_rows = np.sum(all_zero_rows)
# total_rows = len(ppi_feats)
# percentage_all_zero = (num_all_zero_rows / total_rows) * 100
# print(f"Total rows: {total_rows}")
# print(f"Rows with all-zero ppi_feats: {num_all_zero_rows}")
# print(f"Percentage of rows with all-zero ppi_feats: {percentage_all_zero:.2f}%") # 42.52%

####################################################################################################################
with open('data/ppi/ppi-class_map.json', 'r') as file:
    ppi_class_map = json.load(file)

# print(type(ppi_class_map)) #dict keys are the nodes as strings, values are the class labels of len 121
# print(len(ppi_class_map)) # 56944

# print(ppi_class_map["90"])
# print(len(ppi_class_map["90"])) 

# print(ppi_class_map["5000"])
# print(len(ppi_class_map["5000"]))  # each class label is 121

# gene ontology sets as labels (121 in total),

####################################################################################################################

