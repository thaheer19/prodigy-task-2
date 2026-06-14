import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_titanic_eda():
    csv_file = "train.csv"
    
    # 1. LOAD DATASET
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Place the script in the folder with your downloaded data!")
        return
        
    df = pd.read_csv(csv_file)
    print("--- Dataset Loaded Successfully ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    
    # 2. DATA CLEANING (Checking and handling missing values)
    print("--- Missing Values Before Cleaning ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Fill missing values in 'Age' with the median age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill missing values in 'Embarked' with the most common destination port
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Drop 'Cabin' column due to an excessive amount of missing values
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
        
    print("\n--- Cleaning Complete! Remaining Missing Values ---")
    print(df.isnull().sum().sum())
    
    # 3. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Survival Rate Distribution (Target Variable)
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='Survived', hue='Survived', palette='Set1', legend=False)
    plt.title('Distribution of Survival (0 = No, 1 = Yes)')
    plt.xlabel('Survived')
    plt.ylabel('Passenger Count')
    plt.tight_layout()
    plt.savefig('titanic_survival_distribution.png', dpi=300)
    plt.show()

    # Chart 2: Survival Rate split by Passenger Gender/Sex (Relationship Analysis)
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x='Sex', y='Survived', hue='Sex', palette='pastel', legend=False)
    plt.title('Survival Rate by Gender')
    plt.ylabel('Survival Probability')
    plt.tight_layout()
    plt.savefig('titanic_survival_by_gender.png', dpi=300)
    plt.show()

    # Chart 3: Age Distribution of Passengers by Survival Status
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, element='step', palette='muted')
    plt.title('Age Distribution Separated by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('titanic_age_distribution.png', dpi=300)
    plt.show()
    
    print("\nAnalysis complete! Visual charts saved to your project directory.")

if __name__ == "__main__":
    perform_titanic_eda()
