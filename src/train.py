import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.logger import logger
from src.config import MODEL_DIR, NUM_CLASSES, EPOCHS
from src.model import create_or_load_base_model

# =========================================================
# 1️⃣ Load images from the dataset directory
# =========================================================
def load_data_from_dir(
    data_dir="CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    val_split=0.2,
    img_size=(224, 224),
    batch_size=16,
    seed=42
):
    """
    Load image datasets from a directory with subfolders for each class:
        Normal, Cyst, Tumor, Stone
    """
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Extract labels for class weight computation
    y_train = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)

    return train_dataset, val_dataset, y_train

# =========================================================
# 2️⃣ Training function
# =========================================================
def train(data_dir="CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"):
    try:
        logger.info("===== TRAINING STARTED =====")

        # Load datasets
        train_dataset, val_dataset, y_train = load_data_from_dir(data_dir)
        logger.info(f"Train samples: {len(y_train)}")

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        logger.info(f"Class weights: {class_weights_dict}")

        # Create / load model
        model = create_or_load_base_model()
        logger.info("Model ready")
        model.summary(print_fn=logger.info)

        # =====================================================
        # Phase 1: Frozen base
        # =====================================================
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]

        logger.info("Phase 1: Training with frozen base...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=min(EPOCHS, 5),
            class_weight=class_weights_dict,
            callbacks=callbacks,
            verbose=1
        )

        # =====================================================
        # Phase 2: Fine-tuning
        # =====================================================
        logger.info("Phase 2: Fine-tuning...")
        base_model = model.layers[2]  # EfficientNet base layer
        base_model.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_fine = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=3,
            class_weight=class_weights_dict,
            verbose=1
        )

        # Save final model
        final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
        model.save(final_model_path)
        logger.info(f"Final model saved at: {final_model_path}")
        logger.info("===== TRAINING COMPLETED =====")

        return history, history_fine

    except Exception as e:
        logger.exception("Error during training")
        raise e

# =========================================================
# 3️⃣ Run training
# =========================================================
