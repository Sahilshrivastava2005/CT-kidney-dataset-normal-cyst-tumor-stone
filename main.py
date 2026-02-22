import os

from src.logger import logger
from src.config import PROCESSED_DATA_PATH
from src.data_loader import preprocess_data
from src.model import create_or_load_base_model
from src.train import train


def main():
    try:
        logger.info("========== APPLICATION STARTED ==========")

        # =====================================================
        # 1️⃣ Preprocess Dataset (Only if not already processed)
        # =====================================================
        raw_path = os.path.join("data", "raw", "kidneyData.csv")
        processed_path = PROCESSED_DATA_PATH

        if not os.path.exists(processed_path):
            logger.info("Processed dataset not found. Starting preprocessing...")
            preprocess_data(raw_path, processed_path)
        else:
            logger.info("Processed dataset already exists. Skipping preprocessing.")

        # =====================================================
        # 2️⃣ Create or Load Model
        # =====================================================
        model = create_or_load_base_model()
        model.summary()

        # =====================================================
        # 3️⃣ Start Training
        # =====================================================
        logger.info("Starting training pipeline...")
        train()

        logger.info("========== APPLICATION FINISHED SUCCESSFULLY ==========")

    except Exception as e:
        logger.exception("Application crashed")
        raise e


if __name__ == "__main__":
    main()
