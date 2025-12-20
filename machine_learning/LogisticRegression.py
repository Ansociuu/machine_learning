#Logistic Regression (Binary Classification)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for _ in range(self.epochs):
            # Forward
            z = np.dot(X, self.w) + self.b
            y_hat = sigmoid(z)

            # Loss
            loss = -np.mean(
                y * np.log(y_hat + 1e-9) +
                (1 - y) * np.log(1 - y_hat + 1e-9)
            )
            self.losses.append(loss)

            # Gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = np.mean(y_hat - y)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_hat = sigmoid(z)
        return (y_hat >= 0.5).astype(int)
