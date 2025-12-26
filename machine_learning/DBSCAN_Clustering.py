#DBSCAN
import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _region_query(self, X, idx):
        neighbors = []
        for i in range(len(X)):
            if self._distance(X[idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit(self, X):
        n_samples = len(X)
        self.labels = np.full(n_samples, -1)  # -1 là noise
        visited = np.zeros(n_samples)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = 1
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        self.labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            point = neighbors[i]

            if not visited[point]:
                visited[point] = 1
                point_neighbors = self._region_query(X, point)
                if len(point_neighbors) >= self.min_samples:
                    neighbors += point_neighbors

            if self.labels[point] == -1:
                self.labels[point] = cluster_id
            i += 1
#Test DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(X)

plt.scatter(X[:,0], X[:,1], c=dbscan.labels)
plt.show()
