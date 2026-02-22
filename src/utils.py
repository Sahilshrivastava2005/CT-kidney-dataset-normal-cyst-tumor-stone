import pickle
from src.logger import logger
from src.exception import CustomException
import sys

def save_object(file_path, obj):
    try:
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info("Object loaded successfully")
        return obj

    except Exception as e:
        raise CustomException(e, sys)