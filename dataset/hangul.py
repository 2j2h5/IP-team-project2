import os
import numpy as np
import pandas as pd
from PIL import Image

def load_hangul(data_dir='hangul', img_dir='images', csv_file='labels.csv', img_size=(28, 28), normalize=True, flatten=True, one_hot_label=False):
    # Define paths
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = dataset_dir + "/" + data_dir + "/" + csv_file
    img_dir = dataset_dir + "/" + data_dir + "/" + img_dir

    # Load labels from CSV
    labels_df = pd.read_csv(csv_path, header=None, names=['label'])
    img_labels = labels_df['label'].values

    # Generate image paths assuming filenames are in sequence
    num_images = len(img_labels)
    img_paths = [os.path.join(img_dir, f"hangul_{i+1}.jpeg") for i in range(num_images)]

    # Load images and resize
    img_data = np.zeros((num_images, *img_size), dtype=np.float32)

    for idx, img_path in enumerate(img_paths):
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize(img_size)  # Resize image
            img_data[idx] = np.array(img, dtype=np.float32)

    # Normalize image data if needed
    if normalize:
        img_data = img_data.astype(np.float32)
        img_data /= 255.0

    # Reshape image data for compatibility with the network
    if not flatten:
        img_data = img_data.reshape(num_images, 1, *img_size)
    else:
        img_data = img_data.reshape(num_images, -1)

    # Convert labels to one-hot encoding if needed
    if one_hot_label:
        unique_labels = np.unique(img_labels)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        labels = np.zeros((img_labels.size, num_classes))
        for idx, label in enumerate(img_labels):
            labels[idx][label_to_index[label]] = 1
    else:
        labels = img_labels

    # Split into training and testing datasets
    indices = np.arange(num_images)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(num_images * 0.8)  # Use 80% of data for training
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    x_train = img_data[train_indices]
    t_train = labels[train_indices]
    x_test = img_data[test_indices]
    t_test = labels[test_indices]

    t_train = t_train.astype(np.uint8)
    t_test = t_test.astype(np.uint8)

    return (x_train, t_train), (x_test, t_test)