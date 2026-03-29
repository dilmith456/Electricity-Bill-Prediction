import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import FileOperationError
from src.log import logging
from src.utils import save_object


# -----------------------------
# Config Class
# -----------------------------

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "model_metrics.txt")


# -----------------------------
# Model Trainer Class
# -----------------------------

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing data")

            # -----------------------------
            # Split features and target
            # -----------------------------
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # -----------------------------
            #  CRITICAL FIX (FORCE NUMERIC)
            # -----------------------------
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

            # -----------------------------
            # DEBUG CHECKS (VERY IMPORTANT)
            # -----------------------------
            print("\nDEBUG INFO:")
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("y_train sample:", y_train[:5])
            print("y_test sample:", y_test[:5])

            # -----------------------------
            # Improved Random Forest Model
            # -----------------------------
            logging.info("Training Tuned RandomForest model")

            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # -----------------------------
            # Train model
            # -----------------------------
            model.fit(X_train, y_train)

            # -----------------------------
            # Predictions
            # -----------------------------
            y_pred = model.predict(X_test)

            # -----------------------------
            # Evaluation Metrics
            # -----------------------------
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            metrics = {
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            }

            print("\nModel Performance:")
            print(metrics)

            logging.info(f"Model Performance: {metrics}")

            # -----------------------------
            # Save model
            # -----------------------------
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=model
            )

            logging.info("Model saved successfully")

            # -----------------------------
            # Save metrics
            # -----------------------------
            with open(self.config.metrics_file_path, "w") as f:
                f.write(str(metrics))

            logging.info("Metrics saved successfully")

            return metrics

        except Exception as e:
            raise FileOperationError(e, sys)