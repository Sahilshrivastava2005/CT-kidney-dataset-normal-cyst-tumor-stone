import os
import tensorflow as tf

from src.logger import logger
from src.config import MODEL_DIR, IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE


def create_or_load_base_model(
    model_name="base_model.keras"
):
    """
    Creates or loads EfficientNetB0 based classification model.
    """

    save_path = os.path.join(MODEL_DIR, model_name)

    # =====================================================
    # 1️⃣ Load Existing Model
    # =====================================================
    if os.path.exists(save_path):
        logger.info("Loading existing base model...")
        model = tf.keras.models.load_model(save_path)
        return model

    logger.info("Creating new EfficientNetB0 base model...")

    # =====================================================
    # 2️⃣ EfficientNet Base
    # =====================================================
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    base_model.trainable = False  # Freeze base model initially

    # =====================================================
    # 3️⃣ Build Custom Head
    # =====================================================
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # EfficientNet preprocessing
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    # =====================================================
    # 4️⃣ Compile Model
    # =====================================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =====================================================
    # 5️⃣ Save Model
    # =====================================================
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Saving base model...")
    model.save(save_path)
    logger.info("Base model saved successfully")

    return model
