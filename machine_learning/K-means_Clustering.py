#K-means
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def fit(self, X):
        n_samples, n_features = X.shape

        # 1. Khởi tạo centroid ngẫu nhiên từ dữ liệu
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # 2. Gán nhãn cho mỗi điểm
            self.labels = []
            for x in X:
                distances = [self._distance(x, c) for c in self.centroids]
                self.labels.append(np.argmin(distances))
            self.labels = np.array(self.labels)

            # 3. Cập nhật centroid
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))
            new_centroids = np.array(new_centroids)

            # 4. Kiểm tra hội tụ
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._distance(x, c) for c in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)
#Test K-means
import matplotlib.pyplot as plt

# Tạo dữ liệu giả
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) + [0, 5]
])

kmeans = KMeans(k=3)
kmeans.fit(X)

plt.scatter(X[:,0], X[:,1], c=kmeans.labels)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c='red', marker='x')
plt.show()

