# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

    def handle_missing_values(self, df):
        """
        Use Median for skewed data (like Rooms).
        """
        # We only apply imputer to numeric columns with missing values
        imputer = SimpleImputer(strategy='median')
        df['RM'] = imputer.fit_transform(df[['RM']])
        return df

    def scale_features(self, df):
        """
        Scaling ensures that features with large ranges (like TAX) 
        don't dominate features with small ranges (like NOX).
        """
        scaler = StandardScaler()
        # We scale everything except the target 'MEDV'
        features = df.drop(columns=['MEDV'])
        target = df['MEDV']
        
        scaled_features = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
        df_scaled['MEDV'] = target.values
        return df_scaled

    def initiate_preprocessing(self):
        print("Starting Preprocessing...")
        
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # 1. Handle Missing Values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)

        # 2. Scaling (Important for models like Linear Regression or SVM)
        train_df = self.scale_features(train_df)
        test_df = self.scale_features(test_df)

        # 3. Save processed data
        train_df.to_csv(f"{self.processed_dir}/train_processed.csv", index=False)
        test_df.to_csv(f"{self.processed_dir}/test_processed.csv", index=False)
        
        print("Preprocessing Complete! Scaled data saved to 'data/processed/'.")
        return f"{self.processed_dir}/train_processed.csv", f"{self.processed_dir}/test_processed.csv"

if __name__ == "__main__":
    # Assuming ingestion was run and files are in data/
    preprocessor = DataPreprocessor("data/train.csv", "data/test.csv")
    preprocessor.initiate_preprocessing()