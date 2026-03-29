import os
import sys
import joblib

from src.exception import FileOperationError


# Define a function that will save the object
def save_object(file_path, obj):
    try:
        # Get the directory path
        dir_path = os.path.dirname(file_path)

        # If the file path does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object in the file
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise FileOperationError(e, sys)