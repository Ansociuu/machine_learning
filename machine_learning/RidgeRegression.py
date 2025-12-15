import numpy as np

class RidgeRegressionManual:
    def __init__(self, learning_rate=0.01, epochs=1000, lmbd=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lmbd = lmbd  # lambda: regularization parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Dự đoán
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Gradient với regularization L2
            dw = (1/n_samples) * (np.dot(X.T, error) + self.lmbd * self.weights)
            db = (1/n_samples) * np.sum(error)

            # Cập nhật weights và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
