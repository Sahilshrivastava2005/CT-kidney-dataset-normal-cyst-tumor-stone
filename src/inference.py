import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.logger import logger
from src.config import MODEL_DIR, IMAGE_SIZE


# =====================================================
# Class Names (IMPORTANT)
# =====================================================
CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Tumor",
    "Stone"
]


def load_model():
    model_path = os.path.join(MODEL_DIR, "best_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model.keras not found")

    logger.info("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    return model


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    return image


def predict(image_path):
    try:
        model = load_model()

        logger.info(f"Running inference on: {image_path}")

        image = preprocess_image(image_path)

        preds = model.predict(image)
        preds = np.squeeze(preds)

        predicted_index = np.argmax(preds)
        confidence = preds[predicted_index]

        predicted_class = CLASS_NAMES[predicted_index]

        print("\n==============================")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {confidence:.4f}")
        print("==============================\n")

        # Display image
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f})")
        plt.axis("off")
        plt.show()

    except Exception as e:
        logger.exception("Error during inference")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    predict(args.image)
