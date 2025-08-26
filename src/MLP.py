import numpy as np

class MLP:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.activation_func = lambda x: np.maximum(0, x)
        self.softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    def predict(self, x):
        if np.sum(x) == 0.0:
            return np.zeros(10)
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                x = self.activation_func(x @ self.weights[i] + self.biases[i])
            else:
                x = x @ self.weights[i] + self.biases[i]
        
        x = self.softmax(x)
        return x