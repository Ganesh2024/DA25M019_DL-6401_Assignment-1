"""
Data Loader Module
Handles downloading, loading, and preprocessing of MNIST and Fashion-MNIST datasets.
"""

import numpy as np


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_data(dataset: str = "mnist"):
    """
    Load train/validation/test splits for MNIST or Fashion-MNIST.

    Parameters
    ----------
    dataset : 'mnist' | 'fashion'

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    All X arrays have shape (N, 784), float32, values in [0, 1].
    All y arrays have shape (N,), int32.
    """
    dataset = dataset.lower()
    if dataset not in ("mnist", "fashion"):
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion'.")

    (X_train_full, y_train_full), (X_test, y_test) = _keras_load(dataset)

    # Preprocess
    X_train_full = preprocess(X_train_full)
    X_test       = preprocess(X_test)
    y_train_full = y_train_full.astype(np.int32)
    y_test       = y_test.astype(np.int32)

    # Split last 10 % of training data as validation set
    val_size    = int(0.1 * len(X_train_full))
    X_val       = X_train_full[-val_size:]
    y_val       = y_train_full[-val_size:]
    X_train     = X_train_full[:-val_size]
    y_train     = y_train_full[:-val_size]

    print(
        f"[DataLoader] {dataset}  "
        f"train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess(images: np.ndarray) -> np.ndarray:
    """
    Flatten and normalise raw image arrays.

    Parameters
    ----------
    images : np.ndarray, shape (N, 28, 28) uint8

    Returns
    -------
    np.ndarray, shape (N, 784) float32 in [0, 1]
    """
    N = images.shape[0]
    return images.reshape(N, -1).astype(np.float32) / 255.0


# -----------------------------------------------------------------------
# Dataset-specific helpers
# -----------------------------------------------------------------------

def _keras_load(dataset: str):
    """
    Use Keras (TF) to download and return raw numpy arrays.
    Falls back to a manual download if Keras is unavailable.
    """
    try:
        if dataset == "mnist":
            from keras.datasets import mnist as ds
        else:
            from keras.datasets import fashion_mnist as ds
        return ds.load_data()
    except ImportError:
        # Fallback: manual download via urllib
        return _manual_load(dataset)


def _manual_load(dataset: str):
    """
    Manual download of MNIST/Fashion-MNIST as a last resort.
    Saves cached .npy files to ./data/ to avoid repeated downloads.
    """
    import os
    import urllib.request
    import gzip
    import struct

    urls = {
        "mnist": {
            "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "test_images":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        },
        "fashion": {
            "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        },
    }

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    def download_and_parse(url, kind):
        filename = os.path.join(cache_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, filename)
        with gzip.open(filename, "rb") as f:
            if kind == "images":
                _, n, r, c = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
            else:
                _ = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    ds_urls = urls[dataset]
    X_train = download_and_parse(ds_urls["train_images"], "images")
    y_train = download_and_parse(ds_urls["train_labels"], "labels")
    X_test  = download_and_parse(ds_urls["test_images"],  "images")
    y_test  = download_and_parse(ds_urls["test_labels"],  "labels")

    return (X_train, y_train), (X_test, y_test)


# -----------------------------------------------------------------------
# Class name mappings (for visualisation)
# -----------------------------------------------------------------------

CLASS_NAMES = {
    "mnist":   [str(i) for i in range(10)],
    "fashion": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
}