import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import FileOperationError
from src.log import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# -----------------------------
# Config Class
# -----------------------------

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.joblib")


# -----------------------------
# Data Transformation Class
# -----------------------------

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ----------------------------------
    # Create Preprocessor Object
    # ----------------------------------

    def get_data_transformer_obj(self):
        try:
            # Numerical Features
            numerical_columns = [
                'household_size',
                'inflation_rate',
                'electricity_tariff_rate',
                'past_month_units',
                'ac_hours',
                'fan_hours',
                'tv_hours',
                'fridge_hours'
            ]

            # Categorical Features
            categorical_columns = [
                'location_type',
                'climate_zone',
                'income_level'
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    
                ]
            )

            logging.info("Pipelines created successfully")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise FileOperationError(e, sys)

    # ----------------------------------
    # Initiate Data Transformation
    # ----------------------------------

    def initiate_data_transformer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # -----------------------------
            # FORCE CLEAN (CRITICAL FIX)
            # -----------------------------
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            print("Train NaN check:\n", train_df.isnull().sum())
            print("Test NaN check:\n", test_df.isnull().sum())

            logging.info("Training and Testing datasets loaded")

            target_col = "next_month_units"
            columns_to_drop = ["date", "household_id"]

            # -----------------------------
            # REQUIRED COLUMNS (FIXED)
            # -----------------------------
            required_columns = set([
                target_col,
                "household_size",
                "inflation_rate",
                "electricity_tariff_rate",
                "past_month_units",
                "ac_hours",
                "fan_hours",
                "tv_hours",
                "fridge_hours",
                "location_type",
                "climate_zone",
                "income_level",
                *columns_to_drop,
            ])

            missing_train_columns = sorted(required_columns - set(train_df.columns))
            missing_test_columns = sorted(required_columns - set(test_df.columns))

            if missing_train_columns:
                raise ValueError(
                    f"Train dataset is missing required columns: {missing_train_columns}"
                )

            if missing_test_columns:
                raise ValueError(
                    f"Test dataset is missing required columns: {missing_test_columns}"
                )

            # -----------------------------
            # TARGET
            # -----------------------------
            training_target_feature = train_df[target_col]
            testing_target_feature = test_df[target_col]

            # -----------------------------
            # ALIGN DATA (VERY IMPORTANT)
            # -----------------------------
            train_df = train_df.loc[training_target_feature.index]
            test_df = test_df.loc[testing_target_feature.index]

            # -----------------------------
            # INPUT FEATURES
            # -----------------------------
            training_input_features = train_df.drop(columns=[target_col, *columns_to_drop])
            testing_input_features = test_df.drop(columns=[target_col, *columns_to_drop])

            # -----------------------------
            # PREPROCESSOR
            # -----------------------------
            preprocessor_obj = self.get_data_transformer_obj()

            logging.info("Applying the preprocessor")

            training_input_features_array = preprocessor_obj.fit_transform(
                training_input_features
            )

            testing_input_features_array = preprocessor_obj.transform(
                testing_input_features
            )

            logging.info("Data preprocessing completed")

            # -----------------------------
            # FINAL ARRAYS
            # -----------------------------
            training_arr = np.c_[
                training_input_features_array,
                np.array(training_target_feature)
            ]

            testing_arr = np.c_[
                testing_input_features_array,
                np.array(testing_target_feature)
            ]

            # -----------------------------
            # SAVE PREPROCESSOR
            # -----------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor object saved successfully")

            return (
                training_arr,
                testing_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise FileOperationError(e, sys)