import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Check for Missing Values
    print("--- Missing Values ---")
    print(df.isnull().sum())

    # 2. Correlation Matrix - To see which features impact Price (MEDV)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig('data/eda_correlation.png')

    # 3. Relationship between Rooms and Price
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='RM', y='MEDV')
    plt.title("Impact of Rooms (RM) on Price (MEDV)")
    plt.savefig('data/eda_rm_vs_price.png')

    print("EDA Visualizations saved successfully.")

if __name__ == "__main__":
    perform_eda("data/data.csv")