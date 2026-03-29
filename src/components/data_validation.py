import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import FileOperationError
from src.log import logging


@dataclass
class DataValidationConfig:
    status_file_path: str = os.path.join("artifacts", "data_validation_status.txt")


class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()

    def validate_data(self, file_path):
        try:
            df = pd.read_csv(file_path)

            logging.info("Starting data validation")

            required_columns = [
                "household_id",
                "date",
                "location_type",
                "climate_zone",
                "household_size",
                "income_level",
                "inflation_rate",
                "electricity_tariff_rate",
                "past_month_units",
                "next_month_units"
            ]

            validation_status = True
            messages = []

            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_status = False
                messages.append(f"Missing columns: {missing_columns}")

            # Check if dataset is empty
            if df.empty:
                validation_status = False
                messages.append("Dataset is empty")

            # Optional: check duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                messages.append(f"Duplicate rows found: {duplicate_count}")

            # Optional: check missing values
            missing_values = df.isnull().sum()
            missing_dict = missing_values[missing_values > 0].to_dict()
            if missing_dict:
                messages.append(f"Missing values found: {missing_dict}")

            # Save validation result
            with open(self.config.status_file_path, "w") as f:
                f.write(f"Validation Status: {validation_status}\n")
                for msg in messages:
                    f.write(msg + "\n")

            logging.info(f"Validation completed. Status: {validation_status}")

            return validation_status

        except Exception as e:
            raise FileOperationError(e, sys)