import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.config import (
    BASE_DIR,
    PROCESSED_DATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    RANDOM_STATE,
    NUM_CLASSES
)

# =========================================================
# 1️⃣ Preprocess Raw CSV (Fix image paths)
# =========================================================

def preprocess_data(input_path, output_path):
    """
    Cleans raw CSV and fixes incorrect image paths.
    Saves cleaned dataset to processed folder.
    """

    try:
        logger.info("Loading raw data...")
        df = pd.read_csv(input_path)

        logger.info("Fixing image paths...")

        # Remove Colab prefix if exists
        df["path"] = df["path"].apply(
            lambda x: x.replace("/content/data/", "")
        )

        # Fix folder naming mismatch
        df["path"] = df["path"].str.replace(
            r"^CT KIDNEY DATASET Normal, CYST, TUMOR and STONE",
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
            regex=True
        )

        logger.info("Creating processed folder if not exists...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info("Saving processed dataset...")
        df.to_csv(output_path, index=False)

        logger.info("Preprocessing completed successfully")
        logger.info(f"Total samples: {len(df)}")

    except Exception as e:
        logger.exception("Error during preprocessing")
        raise e


# =========================================================
# 2️⃣ Image Loading Utilities
# =========================================================

def load_image(image_path, label):
    """
    Reads image from disk and applies preprocessing.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalize

    return image, label


# =========================================================
# 3️⃣ Create tf.data Dataset
# =========================================================

def create_dataset(image_paths, labels, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =========================================================
# 4️⃣ Load Data for Training
# =========================================================

def load_data():
    """
    Loads processed CSV and returns train/val tf.data datasets.
    """

    try:
        logger.info("Loading processed dataset...")

        df = pd.read_csv(PROCESSED_DATA_PATH)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Use correct columns
        IMAGE_COLUMN = "path"
        LABEL_COLUMN = "target"

        # Convert path to absolute path
        df[IMAGE_COLUMN] = df[IMAGE_COLUMN].apply(
            lambda x: os.path.join(BASE_DIR, x)
        )

        # Train-Val split
        train_df, val_df = train_test_split(
            df,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_STATE,
            stratify=df[LABEL_COLUMN]
        )

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")

        # Create datasets
        train_dataset = create_dataset(
            train_df[IMAGE_COLUMN].values,
            train_df[LABEL_COLUMN].values,
            training=True
        )

        val_dataset = create_dataset(
            val_df[IMAGE_COLUMN].values,
            val_df[LABEL_COLUMN].values,
            training=False
        )

        logger.info("Data loading completed successfully")

        return train_dataset, val_dataset, NUM_CLASSES

    except Exception as e:
        logger.exception("Error while loading data")
        raise e
