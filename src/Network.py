import numpy as np

class NeuralNetwork:
    def __init__(self, layers, rng):
        self.layers = layers
        self.weights = [
            rng.normal(loc=0.0, scale=np.sqrt(2 / layers[i]), size=(layers[i], layers[i + 1])).astype(np.float32)
            for i in range(len(layers) - 1)
        ]
        self.biases = [
            np.zeros(layers[i + 1], dtype=np.float32) for i in range(len(layers) - 1)
        ]
        self.activation_func = lambda x: np.maximum(0, x)
        self.softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    def calculate_one_sample(self, x):
        activations = [x]
        z_values = []
        for i in range(len(self.weights)):
            z = x @ self.weights[i] + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:
                x = self.activation_func(z)
                activations.append(x)
            else:
                x = z
                activations.append(x)
        
        x = self.softmax(x)
        return x, activations, z_values
    
    def calculate_loss_one_sample(self, x, y):
        return - y * np.log(self.calculate_one_sample(x)[0] + np.float32(1e-12))
    
    def calculate_gradient_one_sample(self, x, y):
        s, A, Z = self.calculate_one_sample(x)
        Z_gradients = [np.zeros_like(arr) for arr in Z]
        weights_gradients = [np.zeros_like(arr) for arr in self.weights]

        s_gradients = []
        for i in range(0, len(s)):
            s_gradients.append(- y[i] / (s[i] + np.float32(1e-10)))

        e_values = np.exp(A[-1] - np.max(A[-1]))
        e_sum = np.sum(e_values)
        for i in range(0, len(Z_gradients[-1])):            
            for j in range(0, len(s)):
                if i == j:
                    Z_gradients[-1][i] += s[j] * ((e_sum - e_values[i])/e_sum) * s_gradients[j]
                
                else:
                    Z_gradients[-1][i] -= s[j] * (e_values[i]/e_sum) * s_gradients[j]

            for j in range(len(weights_gradients[-1])):
                weights_gradients[-1][j][i] = Z_gradients[-1][i] * A[-2][j]
        
        for i in range(2, len(Z_gradients) + 1):
            for j in range(0, len(Z_gradients[-i])):
                if Z[-i][j] <= 0:
                    Z_gradients[-i][j] = 0
                    continue
                
                Z_gradients[-i][j] = np.dot(self.weights[-i + 1][j], Z_gradients[-i + 1])
                weights_gradients[-i][:, j] = Z_gradients[-i][j] * A[-i - 1]
                    
        return weights_gradients, Z_gradients

    def train(self, X, Y, epochs, learning_rate, b1, b2, alpha, batch_size, calc_accuracy):
        learning_rate  = np.float32(learning_rate)
        b1 = np.float32(b1)
        b2 = np.float32(b2)
        eps = np.float32(1e-8)

        n = len(X)
        m_w, v_w = [np.zeros_like(arr) for arr in self.weights], [np.zeros_like(arr) for arr in self.weights]
        m_b, v_b = [np.zeros_like(arr) for arr in self.biases], [np.zeros_like(arr) for arr in self.biases]

        t = 0

        best_accuracy = 0.1
        best_weights = None
        best_biases = None

        for i in range(epochs):
            print(f"Epoch: {i}")
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = perm[start:start+batch_size]

                cumulative_w_gradient = [np.zeros_like(W) for W in self.weights]
                cumulative_b_gradient = [np.zeros_like(b) for b in self.biases]

                for j in batch_idx:
                    dW, db = self.calculate_gradient_one_sample(X[j], Y[j])
                    for k in range(len(self.weights)):
                        cumulative_w_gradient[k] += dW[k]
                        cumulative_b_gradient[k] += db[k]

                average_w_gradient = [grad / len(batch_idx) for grad in cumulative_w_gradient]
                average_b_gradient = [grad / len(batch_idx) for grad in cumulative_b_gradient]

                t += 1
                for k in range(len(self.weights)):
                    # m, v - Adam
                    m_w[k] = b1 * m_w[k] + (np.float32(1.0) - b1) * average_w_gradient[k]
                    v_w[k] = b2 * v_w[k] + (np.float32(1.0) - b2) * (average_w_gradient[k] * average_w_gradient[k])

                    m_b[k] = b1 * m_b[k] + (np.float32(1.0) - b1) * average_b_gradient[k]
                    v_b[k] = b2 * v_b[k] + (np.float32(1.0) - b2) * (average_b_gradient[k] * average_b_gradient[k])

                    m_w_hat = m_w[k] / (np.float32(1.0) - (b1 ** t))
                    v_w_hat = v_w[k] / (np.float32(1.0) - (b2 ** t))
                    m_b_hat = m_b[k] / (np.float32(1.0) - (b1 ** t))
                    v_b_hat = v_b[k] / (np.float32(1.0) - (b2 ** t))

                    self.weights[k] *= (1 - learning_rate * alpha) # L2 Regularization
                    self.weights[k] -= learning_rate * (m_w_hat / (np.sqrt(v_w_hat) + eps))
                    self.biases[k] -= learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + eps))
            
                current_accuracy = calc_accuracy(self)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    print(f"New best accuracy {best_accuracy}")
                    best_weights = [w.copy() for w in self.weights]
                    best_biases  = [b.copy() for b in self.biases]

        self.weights = best_weights
        self.biases = best_biases
    
    def calculate_naive_gradient(self, x, y, h):
        w_gradients = [np.zeros_like(arr) for arr in self.weights]
        b_gradients = [np.zeros_like(arr) for arr in self.biases]
        for i in range(len(w_gradients)):
            for j in range(len(w_gradients[i])):
                for k in range(len(w_gradients[i][j])):
                    tmp = self.weights[i][j][k]
                    self.weights[i][j][k] = tmp + h
                    left = np.sum(self.calculate_loss_one_sample(x, y))
                    self.weights[i][j][k] = tmp - h
                    right = np.sum(self.calculate_loss_one_sample(x, y))
                    self.weights[i][j][k] = tmp
                    w_gradients[i][j][k] = (left - right) / (2*h)
        
        for i in range(len(b_gradients)):
            for j in range(len(b_gradients[i])):
                tmp = self.biases[i][j]
                self.biases[i][j] = tmp + h
                left = np.sum(self.calculate_loss_one_sample(x, y))
                self.biases[i][j] = tmp - h
                right = np.sum(self.calculate_loss_one_sample(x, y))
                self.biases[i][j] = tmp
                b_gradients[i][j] = (left - right) / (2*h)
        
        return w_gradients, b_gradients