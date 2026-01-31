# model_trainer.py (Fixed Pipeline Structure)
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error

class ModelTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = "artifacts"
        os.makedirs(self.model_dir, exist_ok=True)

    def initiate_model_trainer(self):
        print("Starting Model Training...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        X_train = train_df.drop(columns=['MEDV'])
        y_train = train_df['MEDV']
        X_test = test_df.drop(columns=['MEDV'])
        y_test = test_df['MEDV']

        # 1. Define feature groups
        stubborn_cols = ['CRIM', 'ZN', 'B']
        # All other columns
        other_cols = [col for col in X_train.columns if col not in stubborn_cols]

        # This way, each column is cleaned AND transformed in one go
        stubborn_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('quantile', QuantileTransformer(output_distribution='normal', random_state=42))
        ])

        standard_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        # 3. Combine them in ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('stubborn', stubborn_transformer, stubborn_cols),
                ('standard', standard_transformer, other_cols)
            ]
        )

        # 4. Final Pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])

        # Train
        model_pipeline.fit(X_train, y_train)

        # Evaluate
        predictions = model_pipeline.predict(X_test)
        print(f"Model R2 Score: {r2_score(y_test, predictions):.4f}")
        
        # Save
        model_file = os.path.join(self.model_dir, "house_price_pipeline.pkl")
        joblib.dump(model_pipeline, model_file)
        print(f"Pipeline saved to {model_file}")
        return model_file

if __name__ == "__main__":
    trainer = ModelTrainer("data/processed/train_engineered.csv", "data/processed/test_engineered.csv")
    trainer.initiate_model_trainer()