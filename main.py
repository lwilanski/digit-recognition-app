from src.Network import NeuralNetwork
import numpy as np
import os
from src.utils.files import read_labels, read_images
from src.utils.preprocess import preprocess_images, one_hot_encode
from src.utils.evaluate import train_and_save, load_weights_biases, evaluate_model_accuracy
from src.utils.images import interactive_mnist_demo, show_image
from collections import Counter

layers = [400, 512, 10]
# rng = np.random.default_rng(seed=15)
rng = np.random.default_rng(seed=47)

dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset/MNIST_ORG')

train_images_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')

X_train_raw = read_images(train_images_path)
y_train_raw = read_labels(train_labels_path)

X_train = preprocess_images(X_train_raw)
y_train = one_hot_encode(y_train_raw)

test_images_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')

X_test_raw = read_images(test_images_path)
y_test_raw = read_labels(test_labels_path)

X_test = preprocess_images(X_test_raw)
y_test = one_hot_encode(y_test_raw)

net, train_pr, test_pr = train_and_save(
    layers=layers, 
    rng=rng, 
    X_train=X_train, 
    y_train=y_train, 
    epochs=8, 
    lr=0.001, 
    b1=0.9, 
    b2=0.99,
    alpha=0.1, 
    batch_size=128, 
    X_test=X_test, 
    y_test=y_test
)

# net = NeuralNetwork(layers, rng)
# w, b = load_weights_biases("hl512_ep16_lr0.001_b10.90_b20.990_bs128_train99.94_test98.34")

# net.weights = w
# net.biases = b

# train_accuracy = evaluate_model_accuracy(net, X_train, y_train)
# test_accuracy = evaluate_model_accuracy(net, X_test, y_test)
# interactive_mnist_demo(net, X_test_raw, X_test, y_test)

# net = NeuralNetwork([3, 4, 5, 3], rng)

# test_sample = np.array([4.5, 1.2, -6.7])
# ground_truth = np.array([0.0, 1.0, 0.0])

# w_grad, b_grad = net.calculate_naive_gradient(test_sample, ground_truth, 0.000001)

# print("Naive Weights gradients")
# for w in w_grad:
#     print(w)
#     print("\n")

# print("Naive Biases gradients")
# for b in b_grad:
#     print(b)
#     print("\n")

# w_grad, b_grad = net.calculate_gradient_one_sample(test_sample, ground_truth)

# print("BP Weights gradients")
# for w in w_grad:
#     print(w)
#     print("\n")

# print("BP Biases gradients")
# for b in b_grad:
#     print(b)
#     print("\n")