import numpy as np

class LassoRegressionCoordinateDescent:
    def __init__(self, lmbd=0.1, max_iter=1000, tol=1e-4):
        self.lmbd = lmbd    #lmbd: regularization parameter
        self.max_iter = max_iter  #max_iter: số vòng lặp tối đa
        self.tol = tol      #tol: ngưỡng hội tụ
        self.weights = None
        self.bias = None

    def soft_threshold(self, rho, lmbd):
        if rho < -lmbd:
            return rho + lmbd
        elif rho > lmbd:
            return rho - lmbd
        else:
            return 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)  # khởi tạo bias bằng trung bình y
        y_centered = y - self.bias  # trung tâm hóa y

        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            for j in range(n_features):
                # Tính phần còn lại sau khi trừ đóng góp của feature j
                y_pred_except_j = y_centered - X @ self.weights + self.weights[j] * X[:, j]
                rho = np.dot(X[:, j], y_pred_except_j)
                
                # Cập nhật weight j bằng soft-thresholding
                self.weights[j] = self.soft_threshold(rho / n_samples, self.lmbd)
            
            # Kiểm tra hội tụ
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias