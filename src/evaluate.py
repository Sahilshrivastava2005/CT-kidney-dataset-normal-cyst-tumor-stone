import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.logger import logger
from src.config import MODEL_DIR, NUM_CLASSES
from src.data_loader import load_data


def evaluate_model():
    try:
        logger.info("========== EVALUATION STARTED ==========")

        # =====================================================
        # 1️⃣ Load Best Model
        # =====================================================
        model_path = os.path.join(MODEL_DIR, "best_model.keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError("best_model.keras not found")

        model = tf.keras.models.load_model(model_path)
        logger.info("Best model loaded successfully")

        # =====================================================
        # 2️⃣ Load Validation Data
        # =====================================================
        _, val_dataset, _ = load_data()

        # =====================================================
        # 3️⃣ Evaluate Loss & Accuracy
        # =====================================================
        loss, accuracy = model.evaluate(val_dataset)
        logger.info(f"Validation Loss: {loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")

        # =====================================================
        # 4️⃣ Get Predictions
        # =====================================================
        y_true = []
        y_pred = []

        for images, labels in val_dataset:
            preds = model.predict(images)
            preds = np.argmax(preds, axis=1)

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # =====================================================
        # 5️⃣ Classification Report
        # =====================================================
        report = classification_report(y_true, y_pred)
        logger.info("\n" + report)

        print("\nClassification Report:\n")
        print(report)

        # =====================================================
        # 6️⃣ Confusion Matrix
        # =====================================================
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        logger.info(f"Confusion matrix saved at {cm_path}")

        logger.info("========== EVALUATION COMPLETED ==========")

    except Exception as e:
        logger.exception("Error during evaluation")
        raise e
