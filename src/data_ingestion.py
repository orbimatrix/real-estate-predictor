# data_ingestion.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.ingested_train_path = "data/train.csv"
        self.ingested_test_path = "data/test.csv"

    def initiate_data_ingestion(self):
        print("Starting Data Ingestion...")
        
        # 1. Read the raw data
        df = pd.read_csv(self.raw_data_path)
        
        # 2. Create the data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # 3. Train-Test Split (Industry Standard: 80-20 split)
        # We use random_state for reproducibility
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        # 4. Save to CSVs for the next stage of the pipeline
        train_set.to_csv(self.ingested_train_path, index=False)
        test_set.to_csv(self.ingested_test_path, index=False)

        print(f"Data Ingestion Complete! Files saved in 'data/' folder.")
        return self.ingested_train_path, self.ingested_test_path

if __name__ == "__main__":
    obj = DataIngestion("data/data.csv")
    obj.initiate_data_ingestion()