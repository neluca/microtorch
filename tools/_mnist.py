import os
import requests
import struct
import gzip
import numpy as np


def download_mnist(save_dir: str) -> None:
    base_url = "https://github.com/neluca/my-datasets/tree/main/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(save_dir, file)
        if os.path.exists(file_path):
            continue
        url = base_url + file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file}")
        else:
            print(f"Failed to download {file}. HTTP Response Code: {response.status_code}")


def load_mnist(mnist_dir: str) -> tuple:
    def read_labels(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def read_images(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols, 1)

    train_labels = read_labels(f"{mnist_dir}/train-labels-idx1-ubyte.gz")
    train_images = read_images(f"{mnist_dir}/train-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{mnist_dir}/t10k-labels-idx1-ubyte.gz")
    test_images = read_images(f"{mnist_dir}/t10k-images-idx3-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)
