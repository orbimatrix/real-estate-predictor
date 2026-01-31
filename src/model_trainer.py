# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

class ModelTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = "artifacts"
        os.makedirs(self.model_dir, exist_ok=True)

    def initiate_model_trainer(self):
        print("Starting Model Training...")
        
        # 1. Load the engineered data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # 2. Split Features and Target
        X_train = train_df.drop(columns=['MEDV'])
        y_train = train_df['MEDV']
        X_test = test_df.drop(columns=['MEDV'])
        y_test = test_df['MEDV']

        # 3. Create a Pipeline (The FIX for NaN)
        # The pipeline first fills NaNs, then trains the model
        model_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='median')), # Fixes the NaN error
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])

        # 4. Train the Pipeline
        model_pipeline.fit(X_train, y_train)

        # 5. Evaluate
        predictions = model_pipeline.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        # Inverse log transform to see actual dollar error
        actual_mae = mean_absolute_error(np.expm1(y_test), np.expm1(predictions))

        print(f"Model R2 Score: {r2:.4f}")
        print(f"Mean Absolute Error: ${actual_mae:.2f}k")

        # 6. Save the entire Pipeline
        model_file = os.path.join(self.model_dir, "house_price_pipeline.pkl")
        joblib.dump(model_pipeline, model_file)
        
        print(f"Pipeline saved to {model_file}")
        return model_file

if __name__ == "__main__":
    # Ensure the paths point to your latest engineered CSVs
    trainer = ModelTrainer("data/processed/train_engineered.csv", "data/processed/test_engineered.csv")
    trainer.initiate_model_trainer()


# Model R2 Score: 0.6793
# Mean Absolute Error: $3.17k