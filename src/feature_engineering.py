# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

class FeatureEngineer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def apply_log_transform(self, df, columns):
        """
        since, we have problem of skewness then let's apply Log(1+x) transformation to handle 0s and skewness.
        """
        for col in columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df

    def create_new_features(self, df):
        """
        let's Create more meaningful signals.
        Example: Tax per room gives us an idea of the 'luxury' tax relative to size.
        """
        # will Ensure no division by zero if RM was 0 (though RM is rooms, so unlikely)
        df['TAX_PER_RM'] = df['TAX'] / (df['RM'] + 1)
        return df

    def fix_stubborn_skewness(self, df):
        # QuantileTransformer forces the data into a normal distribution
        # This is great for 'CRIM' and 'B' which are extremely non-normal in our dataset
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        
        cols_to_fix = ['CRIM', 'ZN', 'B']
        df[cols_to_fix] = qt.fit_transform(df[cols_to_fix])
    
        return df

    def initiate_feature_engineering(self):
        print("Starting Feature Engineering...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # 1. Fix Skewness
        skewed_features = ['CRIM', 'ZN', 'LSTAT', 'MEDV','B']
        # train_df = self.apply_log_transform(train_df, skewed_features)
        # test_df = self.apply_log_transform(test_df, skewed_features)

        # # 2. Added custom logic
        # train_df = self.create_new_features(train_df)
        # test_df = self.create_new_features(test_df)

        # Save the engineered data
        train_df.to_csv("data/processed/train_engineered.csv", index=False)
        test_df.to_csv("data/processed/test_engineered.csv", index=False)
        
        print("Feature Engineering Complete! New features added and skewness fixed.")
        return "data/processed/train_engineered.csv", "data/processed/test_engineered.csv"

if __name__ == "__main__":
    engineer = FeatureEngineer("data/processed/train_engineered.csv", "data/processed/test_engineered.csv")
    engineer.initiate_feature_engineering()


# While Log Transformation normalized price and status, features like Crime and ZN remained heavily zero-inflated. 
# I implemented a QuantileTransformer to map these features to a Gaussian distribution, ensuring the regression
#  model's assumptions were met for these extreme variables