#Kian Chryslyr Q. Cadungog
#2BSCS-A

def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

def initialize_centroids(data, k):
    return data[:k]

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

def compute_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            centroids.append(centroid)
        else:
            centroids.append(cluster[0])
    return centroids

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

data = [
    [1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [9, 10],
    [10, 12], [15, 16], [16, 18], [17, 19]
]
k = 2
centroids, clusters = k_means(data, k)
print("Final Centroids:", centroids)
print("Clusters:", clusters)
