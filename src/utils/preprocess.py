import numpy as np
from PIL import Image

def preprocess_images(images: np.ndarray) -> np.ndarray:
    resized_images = np.array([resize_center_image(image, 20) for image in images])
    images_normalized = resized_images.astype(np.float32) / np.float32(255.0)
    images_flattened = images_normalized.reshape(images.shape[0], -1)
    return images_flattened

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    one_hot = np.eye(num_classes)[labels]
    return one_hot.astype(np.float32)

def resize_center_image(image, k):
    n = len(image)
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
