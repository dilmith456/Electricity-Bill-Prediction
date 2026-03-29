import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import FileOperationError
from src.log import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation


# ------------------------------------
# Config Class
# ------------------------------------

@dataclass
class DataIngestionConfig:
    raw_data_file_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_file_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_file_path: str = os.path.join('artifacts', 'test_data.csv')


# ------------------------------------
# Data Ingestion Class
# ------------------------------------

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config if config is not None else DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Commencing data ingestion")

        try:
            data_path = os.path.join("notebook", "train_data.csv")
            df = pd.read_csv(data_path)

            logging.info("Reading data successful")

            required_columns = {"household_id", "date"}
            missing_columns = sorted(required_columns - set(df.columns))

            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {missing_columns}"
                )

            df["date"] = pd.to_datetime(df["date"])

            # ------------------------------------
            # SAVE RAW DATA (NO SORT NEEDED)
            # ------------------------------------
            os.makedirs(os.path.dirname(self.config.raw_data_file_path), exist_ok=True)

            df.to_csv(self.config.raw_data_file_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # ------------------------------------
            # ✅ RANDOM SPLIT (FINAL FIX)
            # ------------------------------------
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )

            logging.info("Random split completed successfully")

            # Save split data
            train_set.to_csv(self.config.train_data_file_path, index=False)
            test_set.to_csv(self.config.test_data_file_path, index=False)

            return (
                self.config.train_data_file_path,
                self.config.test_data_file_path
            )

        except Exception as e:
            logging.error("Data Ingestion Failed")
            raise FileOperationError(e, sys)


# ------------------------------------
# Main Execution
# ------------------------------------

if __name__ == "__main__":

    obj = DataIngestion()

    # STEP 1: Data Ingestion
    train_data, test_data = obj.initiate_data_ingestion()

    # STEP 2: Data Validation
    validator = DataValidation()

    train_status = validator.validate_data(train_data)
    test_status = validator.validate_data(test_data)

    if not train_status or not test_status:
        raise Exception("Data validation failed")

    # STEP 3: Data Transformation
    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_transformer(
        train_data,
        test_data
    )

    # STEP 4: Model Training
    model_trainer = ModelTrainer()

    model_trainer.initiate_model_trainer(train_arr, test_arr)

    logging.info("Full pipeline completed successfully")