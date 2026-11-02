import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load the dataset
url = "https://drive.google.com/uc?export=download&id=13PSqP0oK2rn6OgmuQSoaYwp4PmPn5avX"
data = pd.read_csv(url)

# separate features from the true labels
# we do not use 'class' for training, only for final comparison
x_df = data.drop(columns=['Class'])
y = data['Class'].values
x = x_df.values

# ---------------------------------------------------------
# k-means implementation from scratch
# ---------------------------------------------------------

# define a function for k-means
def kmeans_scratch(data_points, k, max_iters=100):
    # randomly pick initial cluster centers from the data
    np.random.seed(42)
    random_indices = np.random.choice(data_points.shape[0], k, replace=False)
    centroids = data_points[random_indices]
    
    # loop to adjust centers
    for _ in range(max_iters):
        # calculate distance from each point to each center
        distances = np.linalg.norm(data_points[:, np.newaxis] - centroids, axis=2)
        
        # assign points to the closest center
        labels = np.argmin(distances, axis=1)
        
        # calculate new centers by taking the average of assigned points
        new_centroids = np.array([data_points[labels == i].mean(axis=0) for i in range(k)])
        
        # stop if centers do not change anymore
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
        
    return labels, centroids

# finding optimal k using elbow method (wcss)
wcss = []
for k_val in range(1, 11):
    labels, centers = kmeans_scratch(x, k_val)
    # calculate distance to chosen centers
    dist = np.linalg.norm(x[:, np.newaxis] - centers, axis=2)
    # square the distances and sum them up
    min_dist = np.min(dist, axis=1)
    wcss.append(np.sum(min_dist**2))

print("--- finding optimal k ---")
for k_val, val in enumerate(wcss, 1):
    print(f"k={k_val}: {val:.2f}")
print("notice the 'elbow' (biggest drop slows down) is around k=3.\n")

# run final k-means with k=3
kmeans_labels, final_centers = kmeans_scratch(x, k=3)
print("--- final k-means cluster centers (k=3) ---")
print(final_centers)
print("\n")


# ---------------------------------------------------------
# dbscan implementation from scratch
# ---------------------------------------------------------

# standardize the data first for dbscan (crucial for distance measuring)
# subtract the mean and divide by standard deviation for each column
x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# define a function for dbscan
def dbscan_scratch(data_points, eps, min_samples):
    # create an array to hold labels. -2 means unvisited, -1 means noise.
    labels = np.full(data_points.shape[0], -2)
    cluster_id = 0
    
    # go through every single point
    for i in range(data_points.shape[0]):
        # skip if we have already visited this point
        if labels[i] != -2:
            continue
            
        # find all neighbors within eps distance
        distances = np.linalg.norm(data_points - data_points[i], axis=1)
        neighbors = np.where(distances <= eps)[0].tolist()
        
        # if point does not have enough neighbors, label as noise
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            # start a new cluster
            labels[i] = cluster_id
            
            # expand the cluster by checking neighbors
            index = 0
            while index < len(neighbors):
                neighbor_idx = neighbors[index]
                
                # if it was labeled as noise, change it to belong to this cluster
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                    
                # if it is totally unvisited
                elif labels[neighbor_idx] == -2:
                    labels[neighbor_idx] = cluster_id
                    
                    # find neighbors of this neighbor
                    new_distances = np.linalg.norm(data_points - data_points[neighbor_idx], axis=1)
                    new_neighbors = np.where(new_distances <= eps)[0].tolist()
                    
                    # if it is a core point itself, add its neighbors to our checklist
                    if len(new_neighbors) >= min_samples:
                        neighbors.extend(new_neighbors)
                        
                index += 1
                
            # move to next cluster id for the next group
            cluster_id += 1
            
    return labels

# run dbscan with chosen hyperparameters
# eps=0.9 and min_samples=5 work well for this standardized dataset
dbscan_labels = dbscan_scratch(x_scaled, eps=0.9, min_samples=5)

# calculate results
# count how many unique clusters formed (excluding noise which is -1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print("--- dbscan results ---")
print(f"number of clusters formed: {n_clusters}")
print(f"total number of noise/outlier points: {n_noise}\n")


# ---------------------------------------------------------
# visualization and comparison
# ---------------------------------------------------------

# use pca to squeeze the 7 features down to 2 coordinates for plotting
pca = PCA(n_components=2)
x_2d = pca.fit_transform(x_scaled)

# create a wide figure with 3 side-by-side plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. true original labels
axes[0].scatter(x_2d[:, 0], x_2d[:, 1], c=y, cmap='viridis')
axes[0].set_title("true target labels")

# 2. k-means results
axes[1].scatter(x_2d[:, 0], x_2d[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].set_title("k-means clusters (k=3)")

# 3. dbscan results
axes[2].scatter(x_2d[:, 0], x_2d[:, 1], c=dbscan_labels, cmap='viridis')
# distinct red 'x' marks for the noise points
noise_mask = (dbscan_labels == -1)
axes[2].scatter(x_2d[noise_mask, 0], x_2d[noise_mask, 1], c='red', marker='x', label='noise')
axes[2].set_title("dbscan clusters")
axes[2].legend()

plt.show()