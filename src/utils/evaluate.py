import numpy as np
from src.Network import NeuralNetwork
from pathlib import Path
from functools import partial

def train_and_save(layers, rng, X_train, y_train, epochs, lr, b1, b2, alpha, batch_size, X_test, y_test):
    accuracy_func = partial(evaluate_model_accuracy, X_test = X_test, y_test = y_test)
    net = NeuralNetwork(layers, rng)
    net.train(X_train, y_train, epochs, lr, b1, b2, alpha, batch_size, accuracy_func)
    train_accuracy = evaluate_model_accuracy(net, X_train, y_train)
    test_accuracy = evaluate_model_accuracy(net, X_test, y_test)
    lr_tag = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.3f}"
    hidden = "_".join(map(str, layers[1:-1])) or "nohidden"
    run_dir = Path(f"C:/Users/Lukasz/Documents/GitHub/digit-recognition-app/trained-models/input400_hl{hidden}_ep{epochs}_lr{lr_tag}_b1{b1:.2f}_b2{b2:.3f}_alpha{alpha:.2f}_bs{batch_size}_train{round(100 * train_accuracy, 2)}_test{round(100 * test_accuracy, 2)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "weights.npz", *net.weights)
    np.savez_compressed(run_dir / "biases.npz",  *net.biases)
    return net, train_accuracy, test_accuracy

def load_weights_biases(path):
    run_dir = Path(f"C:/Users/Lukasz/Documents/GitHub/digit-recognition-app/trained-models")

    weights_data = np.load(run_dir/ path / "weights.npz")
    weights = [weights_data[f"arr_{i}"] for i in range(len(weights_data.files))]

    biases_data = np.load(run_dir/ path / "biases.npz")
    biases = [biases_data[f"arr_{i}"] for i in range(len(biases_data.files))]

    return weights, biases

def evaluate_model_accuracy(model, X_test, y_test):
    n = len(X_test)
    if y_test.ndim == 2:
        true_labels = np.argmax(y_test, axis=1)
    else:
        true_labels = y_test.astype(int)

    correct = 0
    for i in range(n):
        scores = model.calculate_one_sample(X_test[i])[0]
        pred = int(np.argmax(scores))
        if pred == true_labels[i]:
            correct += 1
    return correct / n if n else 0.0
