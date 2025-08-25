import numpy as np
import matplotlib.pyplot as plt

def show_image(image):
    """
    Wyświetla obraz jako skalę szarości za pomocą matplotlib.
    Zakłada, że `image` to 2D macierz typu int w zakresie [0, 255].
    """
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

def interactive_mnist_demo(model, X_test_raw, X_test_proc, y_test_onehot):
    """
    Wyświetla losowe próbki ze zbioru testowego MNIST z predykcjami modelu.

    Parametry:
    - model: wytrenowany model z metodą `calculate_one_sample(x)` -> (output, ...)
    - X_test_raw: obrazy 28x28 w skali 0-255 (np.array shape (N, 28, 28), dtype=np.uint8)
    - X_test_proc: obrazy spłaszczone i znormalizowane do [0.0, 1.0] (np.array shape (N, 784))
    - y_test_onehot: etykiety w formacie one-hot (np.array shape (N, 10))
    """
    n = len(X_test_raw)
    while True:
        idx = np.random.randint(0, n)
        image = X_test_raw[idx]
        input_vector = X_test_proc[idx]
        true_label = np.argmax(y_test_onehot[idx])

        # Predykcja modelu
        probs = model.calculate_one_sample(input_vector)[0]
        pred_label = np.argmax(probs)

        print("\n")

        # Wyświetlenie obrazu
        print(f"Sample index: {idx}")
        # Wyświetlenie prawdopodobieństw
        print("Probabilities:")
        for i, p in enumerate(probs):
            print(f"Class {i}: {p:.2f}")
        print(f"\nTrue class: {true_label}")
        print(f"Predicted class: {pred_label}")

        show_image(image)

        print("\n\n")

def vec400_to_img20(vec):
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size != 400:
        raise ValueError(f"Expected 400 values, got {arr.size}")
    img20 = arr.reshape(20, 20)                     # domyślnie order='C'
    img255 = np.clip(np.rint(img20 * 255), 0, 255).astype(np.uint8)
    return img255
