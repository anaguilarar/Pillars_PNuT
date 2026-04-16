"""
One-time script to convert TF checkpoint weights → HDF5 (.h5) format.

Run this in Colab once (TF 2.16+ is fine), then upload the resulting .h5
file to replace the checkpoint files in your S3 bucket.

Usage in Colab:
    !pip install tf-keras -q
    !python convert_weights_to_h5.py
"""
import os
import zipfile
import requests
from io import BytesIO

import tensorflow as tf
try:
    import tf_keras as keras
    from tf_keras import layers
    from tf_keras.models import Model
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model

WEIGHTS_URL = "https://dlmodels-bucket.s3.ap-northeast-1.amazonaws.com/root_detection.zip"
MODEL_NAME = "vgg16_root_detection"
OUTPUT_H5 = "root_detection.h5"


def build_vgg16_model(width=512, height=512, channels=3):
    vggmodel = keras.applications.vgg16.VGG16(
        input_shape=(width, height, channels),
        weights='imagenet',
        include_top=False)

    for layer in vggmodel.layers:
        layer.trainable = False

    # layer[17] = block5_conv3, matching the original training architecture
    x = layers.Dropout(0.3)(vggmodel.layers[17].output)
    x = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

    return Model(vggmodel.input, x)


def download_and_extract(url, extract_to="models"):
    print(f"Downloading {url} ...")
    req = requests.get(url)
    with zipfile.ZipFile(BytesIO(req.content)) as z:
        z.extractall(extract_to)
    folder = os.path.join(extract_to, os.path.basename(url)[:-4])
    print(f"Extracted to: {folder}")
    return folder


def find_checkpoint(folder, model_name):
    index_files = [f for f in os.listdir(folder) if f.endswith('.index') and f.startswith(model_name)]
    if not index_files:
        raise FileNotFoundError(f"No checkpoint found for '{model_name}' in {folder}")
    checkpoint_path = os.path.join(folder, index_files[0][:-6])
    print(f"Found checkpoint: {checkpoint_path}")
    return checkpoint_path


def convert():
    keras.backend.clear_session()

    model_folder = download_and_extract(WEIGHTS_URL)
    checkpoint_path = find_checkpoint(model_folder, MODEL_NAME)

    print("Building model architecture...")
    model = build_vgg16_model()

    print("Loading checkpoint weights...")
    model.load_weights(checkpoint_path)

    print(f"Saving as HDF5: {OUTPUT_H5}")
    model.save_weights(OUTPUT_H5)

    print(f"\nDone. Upload '{OUTPUT_H5}' to your S3 bucket inside a zip named 'root_detection.zip'.")
    print("Then update WEIGHTS_URL in the notebook to point to the new zip.")


if __name__ == "__main__":
    convert()
