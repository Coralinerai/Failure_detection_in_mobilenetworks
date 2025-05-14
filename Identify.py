import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno
import datetime as dt

# Load data
data = pd.read_csv('data/clustered_data.csv')

# Convert Timestamp to datetime format (if needed)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# Separate the data into two clusters
cluster_0 = data[data['Cluster'] == 0]
cluster_1 = data[data['Cluster'] == 1]
# Descriptive statistics for Cluster 0
print("\nDescriptive statistics for Cluster 0:")
for column in cluster_0.columns:
    if cluster_0[column].dtype in ['int64', 'float64']:
        print(f"\n{column}:")
        print(cluster_0[column].describe())
# Descriptive statistics for Cluster 1
print("\nDescriptive statistics for Cluster 1:")
for column in cluster_1.columns:
    if cluster_1[column].dtype in ['int64', 'float64']:
        print(f"\n{column}:")
        print(cluster_1[column].describe())






