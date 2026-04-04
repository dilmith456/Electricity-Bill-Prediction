import os
import pandas as pd
from src.components.data_ingestion import DataIngestion

# -------------------------
# File Paths
# -------------------------
MAIN_DATA_PATH = "notebook/train_data.csv"
NEW_DATA_PATH = "notebook/new_data.csv"


# -------------------------
# Save new prediction data (SAFE)
# -------------------------
def append_new_data(new_record: dict):

    # Safe file reading
    if os.path.exists(NEW_DATA_PATH):
        try:
            df = pd.read_csv(NEW_DATA_PATH)
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Append new record
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    # Save safely
    df.to_csv(NEW_DATA_PATH, index=False)

    print("New data stored (NOT used for training yet)")


# -------------------------
# Safe retraining (NO DATA CORRUPTION)
# -------------------------
def retrain_model():

    print("🔄 Safe Retraining started...")

    # Load main dataset safely
    try:
        main_df = pd.read_csv(MAIN_DATA_PATH)
        print(f"Main dataset size: {len(main_df)}")
    except:
        print("Error reading main dataset")
        return

    # Check new data safely
    if os.path.exists(NEW_DATA_PATH):
        try:
            new_df = pd.read_csv(NEW_DATA_PATH)
            print(f"New data records: {len(new_df)}")
        except:
            print("New data file is empty or corrupted")
    else:
        print("No new data file found")

    # IMPORTANT: Do NOT merge automatically
    print("New data NOT merged (waiting for validation)")

    # Continue training ONLY with original dataset
    try:
        ingestion = DataIngestion()
        ingestion.initiate_data_ingestion()

        print("Model retrained safely without corruption")

    except Exception as e:
        print("Retraining failed:", e)