from src.Network import NeuralNetwork
import numpy as np
import os
from src.utils.files import read_labels, read_images
from src.utils.preprocess import preprocess_images, one_hot_encode
from src.utils.evaluate import train_and_save, load_weights_biases, evaluate_model_accuracy
from src.utils.images import interactive_mnist_demo, show_image, vec400_to_img20
from collections import Counter

layers = [400, 512, 256, 128, 10]
# rng = np.random.default_rng(seed=15)
rng_model = np.random.default_rng(48)   # do inicjalizacji sieci
rng_data  = np.random.default_rng(123)

dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')

add_dataset_dir = os.path.join(os.path.dirname(__file__), 'additional-examples')

add_data_path = os.path.join(add_dataset_dir, 'additional_examples.npz')
data = np.load(add_data_path)
X_extra, y_extra = data["X"], data["y"]

perm = rng_data.permutation(len(X_extra))

X_extra = X_extra[perm]
y_extra = y_extra[perm]

train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte.gz')

X_train_raw = read_images(train_images_path)
y_train_raw = np.concatenate([read_labels(train_labels_path), y_extra], axis=0)

X_train = np.concatenate([preprocess_images(X_train_raw), X_extra], axis=0)
y_train = one_hot_encode(y_train_raw)

test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz')

X_test_raw = read_images(test_images_path)
y_test_raw = np.concatenate([read_labels(test_labels_path), y_extra], axis=0)

X_test = np.concatenate([preprocess_images(X_test_raw), X_extra], axis=0)
y_test = one_hot_encode(y_test_raw)

print(len(X_train), len(y_train), len(X_test), len(y_test))

network = NeuralNetwork(layers, rng_model)
w, b = load_weights_biases("new_data_input400_hl512_256_128_ep4_lr0.001_b10.90_b20.999_alpha0.10_bs128_train99.78_test98.38")

network.weights = w
network.biases = b

net, train_pr, test_pr = train_and_save(
    network=network,
    layers=layers, 
    rng=rng_model, 
    X_train=X_train, 
    y_train=y_train, 
    epochs=4, 
    lr=0.001, 
    b1=0.9, 
    b2=0.999,
    alpha=0.1, 
    batch_size=128, 
    X_test=X_test, 
    y_test=y_test
)

# net = NeuralNetwork(layers, rng_data)
# w, b = load_weights_biases("new_data_input400_hl512_256_128_ep4_lr0.001_b10.90_b20.999_alpha0.10_bs128_train99.78_test98.38")

# net.weights = w
# net.biases = b

# print(evaluate_model_accuracy(net, X_extra, y_extra))

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