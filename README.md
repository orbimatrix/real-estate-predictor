# Real Estate Price Predictor

A machine learning project to predict median home values using exploratory data analysis and regression modeling.

## Project Overview

This project analyzes real estate data to identify key factors influencing home prices and builds a predictive model. The dataset contains 14 features including crime rates, room counts, property taxes, and more.

## Dataset

**Source:** Boston Housing Dataset

**Features (14 total):**
- **CRIM** - Per capita crime rate by town
- **ZN** - Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS** - Proportion of non-retail business acres per town
- **CHAS** - Charles River dummy variable (1 if bounds river, 0 otherwise)
- **NOX** - Nitric oxides concentration (parts per 10 million)
- **RM** - Average number of rooms per dwelling
- **AGE** - Proportion of owner-occupied units built prior to 1940
- **DIS** - Weighted distances to five Boston employment centers
- **RAD** - Index of accessibility to radial highways
- **TAX** - Full-value property-tax rate per $10,000
- **PTRATIO** - Pupil-teacher ratio by town
- **B** - 1000(Bk - 0.63)² where Bk is proportion of blacks by town
- **LSTAT** - % lower status of the population
- **MEDV** - Median value of owner-occupied homes in $1000's (Target)

## Project Structure

```
real-estate-predictor/
├── data/
│   ├── data.csv                    # Raw dataset
│   ├── train.csv                   # Training set
│   ├── test.csv                    # Test set
│   ├── processed/
│   │   ├── train_processed.csv     # Processed training data
│   │   └── test_processed.csv      # Processed test data
│   └── eda.md                      # EDA notes
├── src/
│   ├── eda_comprehensive.ipynb     # Comprehensive EDA notebook
│   ├── eda.py                      # EDA analysis script
│   ├── data_ingestion.py           # Data loading and ingestion
│   ├── preprocessing.py            # Data preprocessing
│   ├── feature_engineering.py      # Feature engineering
│   └── model_trainer.py            # Model training script
├── notebooks/                      # Additional notebooks
├── tests/                          # Unit tests
├── main.py                         # Main entry point
└── README.md                       # This file
```

## Key Findings from EDA

### Data Characteristics
- **Missing Values:** 5 missing values in RM column
- **Skewness Issues:** 
  - CRIM (Crime rate): Right-skewed (5.25) - extreme outliers
  - B: Right-skewed (3.43) - extreme outliers
  - ZN: Right-skewed
  - LSTAT, PTRATIO: Left-skewed

### Correlations with Price (MEDV)
- **RM (Rooms):** +0.67 correlation - Strongest positive relationship
  - More rooms = Higher price
- **LSTAT (Lower status %):** -0.74 correlation - Strong negative relationship

### Data Issues Identified
1. Right-skewed distributions in CRIM and B require log transformation
2. Outliers in crime rate and racial demographics
3. Non-normal target variable distribution requiring transformation

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis using histograms
- Outlier detection with boxplots
- Correlation matrix analysis
- Skewness assessment

### 2. Data Preprocessing
- Handle missing values
- Apply log transformation to skewed features (CRIM, B, ZN)
- Normalize/standardize features for model compatibility

### 3. Feature Engineering
- Identify primary predictive features (RM, LSTAT)
- Create derived features if needed
- Handle categorical variables

### 4. Model Training
- Linear Regression baseline
- Evaluate with appropriate metrics
- Prevent bias from outliers and skewed distributions

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Analysis
1. **Exploratory Analysis:**
   ```bash
   jupyter notebook src/eda_comprehensive.ipynb
   ```

2. **Full Pipeline:**
   ```bash
   python main.py
   ```

## Usage Examples

### Loading Data (in `src/eda_comprehensive.ipynb`)
```python
import pandas as pd
df = pd.read_csv("../data/data.csv")
```
### Streamlit app
```
streamlit run streamlit_app.py
```
and you must run backend fastapi before making prediction on streamlit app.

### Docker files
In streamlit_app.py, change your request URL from http://localhost:8000/predict to http://backend:8000/predict to allow the containers to talk to each other.



### Visualizing Room vs Price Relationship
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.regplot(x='RM', y='MEDV', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Median Home Value in $1000s (MEDV)')
plt.title('Room Count vs House Price with Regression Line')
plt.grid(True, alpha=0.3)
plt.show()
```


## Model Performance

- Target variable transformation recommended due to non-normal distribution
- Log transformation applied to highly skewed features
- Expected improved model performance after addressing skewness

## Files Description

| File | Purpose |
|------|---------|
| `main.py` | Entry point for the full pipeline |
| `src/eda.py` | Automated EDA functions |
| `src/eda_comprehensive.ipynb` | Interactive exploration and visualization |
| `src/data_ingestion.py` | Data loading utilities |
| `src/preprocessing.py` | Data cleaning and transformation |
| `src/feature_engineering.py` | Feature creation and selection |
| `src/model_trainer.py` | Model training and evaluation |

## Next Steps

1. **Apply transformations** to normalize skewed distributions
2. **Train regression models** with processed features
3. **Cross-validate** model performance
4. **Evaluate** using metrics (R², RMSE, MAE)
5. **Hyperparameter tuning** for optimal performance


## Author

Saqib Iqbal

## License

This project uses the Boston Housing Dataset for educational purposes.
