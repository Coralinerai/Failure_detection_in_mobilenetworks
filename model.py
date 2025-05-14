import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('projet/data/cleaned_data.csv')

# Feature Engineering: Create new features  
data['Signal_SNR_Ratio'] = data['Signal Strength (dBm)'] / (data['SNR'] + 1e-9)  # Avoid division by zero
data['Attenuation_SNR_Ratio'] = data['Attenuation'] / (data['SNR'] + 1e-9)

#Finding the optimal k
features = data.select_dtypes(include=['number'])  # All numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
inertia_values = []
for k in range(1, 11):  # Try 1 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_values)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#after trying , the elbow method , we found that our data
kmeans = KMeans(n_clusters=2, max_iter=500, random_state=42, init='k-means++')
kmeans.fit(scaled_features)

data['Cluster'] = kmeans.labels_
print("Cluster Centers (All Features, Scaled):")
print(kmeans.cluster_centers_)

# PCA for Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clusters (PCA Components of All Features)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 5: Evaluate Clustering
sil_score = silhouette_score(pca_components, kmeans.labels_)
db_score = davies_bouldin_score(pca_components, kmeans.labels_)

print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {db_score}")

# Save clustered data
data.to_csv("projet/data/clustered_data.csv", index=False)

import pickle
from sklearn.cluster import KMeans

#saving the model
with open('projet/model/kmeans_model.pkl', 'wb') as f:
    pickle.dump({
        'model': kmeans,
        'scaler': scaler
    }, f)