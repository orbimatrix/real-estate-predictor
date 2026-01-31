import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Data
df = pd.read_csv('data/data.csv')

# 2. Preprocessing & Feature Engineering
# Handle RM missing values
imputer = SimpleImputer(strategy='median')
df['RM'] = imputer.fit_transform(df[['RM']])

# Target and Features
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Log transform the target (for skewed prices)
y_log = np.log1p(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Transformation for stubborn columns
qt = QuantileTransformer(output_distribution='normal', random_state=42)
stubborn_cols = ['CRIM', 'ZN', 'B']
X_train[stubborn_cols] = qt.fit_transform(X_train[stubborn_cols])
X_test[stubborn_cols] = qt.transform(X_test[stubborn_cols])

# Log transform LSTAT in features
X_train['LSTAT'] = np.log1p(X_train['LSTAT'])
X_test['LSTAT'] = np.log1p(X_test['LSTAT'])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training and Comparison
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    # Inverse transform predictions to get actual prices for evaluation
    # Since we used log1p, we use expm1
    actual_preds = np.expm1(predictions)
    actual_y_test = np.expm1(y_test)
    
    r2 = r2_score(y_test, predictions) # R2 on log scale is fine for comparison
    mae = mean_absolute_error(actual_y_test, actual_preds)
    rmse = np.sqrt(mean_squared_error(actual_y_test, actual_preds))
    
    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE ($1000s)": mae,
        "RMSE ($1000s)": rmse
    })

# Convert to DataFrame for display
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print(results_df)


#                Model  R2 Score  MAE ($1000s)  RMSE ($1000s)
# 1      Random Forest  0.756393      2.643600       5.280018
# 2  Gradient Boosting  0.745102      2.699837       5.363971
# 0  Linear Regression  0.474104      3.842475       7.259961

# The best model is Random Forest with an R2 Score of approximately 0.756, MAE of $2.64k, and RMSE of $5.28k.