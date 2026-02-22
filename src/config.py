import os

# =====================================================
# Base Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "kidneyData_processed.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# =====================================================
# Dataset Settings
# =====================================================

VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
NUM_CLASSES = 4

IMAGE_SIZE = (224, 224)   # 🔥 Required for CNN / Transfer Learning

# =====================================================
# Training Settings
# =====================================================

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
