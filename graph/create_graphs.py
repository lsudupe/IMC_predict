# Script for creating graphs from the single_cell_data.csv file
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
from scipy.spatial import KDTree

# Read data
df = pd.read_csv('./data/df.csv')
grouped = df.groupby('Patient ID')

# Create graphs dic
patient_graphs = {}
for patient_id, data in tqdm(grouped, desc="Processing patients"):
    G = nx.Graph()
    coords = data[['Location_Center_X', 'Location_Center_Y']].values
    tree = KDTree(coords)

    # Add nodes with features
    for index, row in data.iterrows():
        G.add_node(index, CD68=row['CD68'], CD3=row['CD3'], CD20=row['CD20'])

    # Use KDTree to find edges within a specified radius
    for idx, point in enumerate(coords):
        indices = tree.query_ball_point(point, r=50)  # r is the radius
        for i in indices:
            if i != idx and not G.has_edge(idx, i):
                dist = np.linalg.norm(point - coords[i])
                G.add_edge(idx, i, weight=dist)

    # Save graphs in dict
    patient_graphs[patient_id] = G

# Save graphs
save_path = "./results/graphs/graphs_dic_norm.pkl"
with open(save_path, "wb") as f:
    pickle.dump(patient_graphs, f)
