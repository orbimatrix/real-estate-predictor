# 🏡 Real Estate Price Prediction: End-to-End ML Pipeline
This project demonstrates a production-grade Machine Learning workflow, moving from raw data exploration to a containerized deployment. It focuses on solving real-world data challenges like extreme skewness, zero-inflation, and feature mismatch in production.

## 📊 Exploratory Data Analysis (EDA) Summary
Before building the model, an exhaustive statistical analysis was performed on the dataset (511 entries, 14 features).

1. Key Data Insights

- Correlation: RM (Average Rooms) showed a strong positive correlation (+0.70) with price, while LSTAT (% Lower Status Population) showed a strong negative correlation (-0.74).

- Missing Values: Identified 5 missing entries in the RM column, which were handled via median imputation to maintain data integrity.

- Anomalies: Discovered a price "cap" at 50 ($50,000), indicating potential data censoring for high-end properties.

2. The Skewness Problem

- Many features did not follow a Normal (Gaussian) distribution, which can bias linear models and reduce the effectiveness of distance-based algorithms.

- Right Skewed (Positive): CRIM (Crime Rate) and ZN (Land Zone) had long right tails.

- Left Skewed (Negative): B (Neighborhood index) was heavily pulled to the left.

- Target Skew: The house prices (MEDV) required log-transformation to stabilize variance.

## 🛠️ Advanced Data Engineering
To transform "messy" data into "model-ready" data, the following industry-standard techniques were used:

1. Quantile Transformation
For "stubborn" features like CRIM (Skewness: 5.25), a standard Log transform was insufficient. I implemented a QuantileTransformer to map these features to a Normal distribution.

- Result: CRIM skewness was reduced from 5.25 to 0.00.

2. Nested Pipelines & ColumnTransformers

- To prevent Data Leakage and handle feature-specific logic, I built a modular Scikit-learn Pipeline:

- Stubborn Branch: Handles Imputation + Quantile Transformation for CRIM, ZN, and B.

- Standard Branch: Handles Median Imputation for remaining features.

- Model: A Random Forest Regressor (100 estimators) was selected as the optimal "brain" due to its ability to handle non-linear relationships.


## 🚀 Deployment & Application Layer

1. FastAPI Backend
The model is served via a REST API. It includes:

- Pydantic Schemas: Ensures strict type checking for incoming house features.

- On-the-fly Engineering: The API automatically calculates derived features like TAX_PER_RM and applies Log1p to LSTAT before prediction.

2. Streamlit Frontend

- An interactive dashboard allows users to adjust house parameters (Rooms, Tax, Crime Rate, etc.) and receive instant price valuations in USD.

3. Dockerization

The entire stack (Backend + Frontend) is containerized using docker-compose, ensuring the application runs identically in any environment.

## 📈 Model Performance
R2 Score: ~0.75 (Explains 75% of price variance)

Mean Absolute Error (MAE): ~$2.66k (Predictions are off by less than $3,000 on average)

