import numpy as np
from pathlib import Path
from PIL import Image

def load_weights_biases():
    run_dir = Path(f"model-parameters")

    weights_data = np.load(run_dir/ "weights.npz")
    weights = [weights_data[f"arr_{i}"] for i in range(len(weights_data.files))]

    biases_data = np.load(run_dir/ "biases.npz")
    biases = [biases_data[f"arr_{i}"] for i in range(len(biases_data.files))]

    return weights, biases

def resize_center_image(image, k):
    n = len(image)
    top_bound = -1
    bottom_bound = -1
    left_bound = -1
    right_bound = -1
    for i in range(n):
        if np.sum(image[i]) > 0:
            top_bound = i - 1
            break
    
    for i in range(n - 1, -1, -1):
        if np.sum(image[i]) > 0:
            bottom_bound = i + 1
            break

    for i in range(n):
        if np.sum(image[:, i]) > 0:
            left_bound = i - 1
            break
    
    for i in range(n - 1, -1, -1):
        if np.sum(image[:, i]) > 0:
            right_bound = i + 1
            break
    
    content = image[top_bound + 1:bottom_bound, left_bound + 1:right_bound]

    height = bottom_bound - top_bound - 1
    width = right_bound - left_bound - 1

    if height > width:
        new_image = np.zeros((height, height), dtype=np.float32)
        left_edge = (height - width) // 2
        new_image[:, left_edge:left_edge + width] = content
    elif width > height:
        new_image = np.zeros((width, width), dtype=np.float32)
        upper_edge = (width - height) // 2
        new_image[upper_edge:upper_edge + height, :] = content
    else:
        new_image = content

    if len(new_image) != k:
        pil_image = Image.fromarray(new_image)

        resized_image = pil_image.resize((k, k), Image.BILINEAR)

        resized_array = np.array(resized_image, dtype=np.float32)
        
        return resized_array

    else:
        return new_image