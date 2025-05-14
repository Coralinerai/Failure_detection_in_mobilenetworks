import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno
import datetime as dt

# Load data
data = pd.read_csv('train.csv')

# Preview the data
print(data.head())

# Check data types and missing values
print(data.info())

# Summary of numeric values
print(data.describe())

# Count missing values per column
print(data.isna().sum())

# Check unique values in a column
for column in data.columns:
    print(f"Unique values in {column}: {data[column].unique()}")

# Removing unnecessary columns
data = data.drop(['Tower ID', 'User ID'], axis=1)

# Handle the 'Timestamp' column
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Check for invalid dates
invalid_dates = data[data['Timestamp'].isna()]
print(f"Invalid dates:\n{invalid_dates}")

# Drop rows with invalid dates
data = data.dropna(subset=['Timestamp'])

# Transform categorical columns to numerical
data['Incoming/Outgoing'] = data['Incoming/Outgoing'].map({'incoming': 0, 'outgoing': 1})
data['Environment'] = data['Environment'].map({'urban': 0, 'home': 1, 'open': 2, 'suburban': 3})
data['Call Type'] = data['Call Type'].map({'voice': 0, 'data': 1})

# Unify units for 'SNR'
data['SNR'] = 10 * np.log10(data['SNR'])

# Replace outliers with the moderate values (e.g., Q1, Q3, or median)
def replace_outliers(df):
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace values below lower bound with Q1 and above upper bound with Q3
        df[col] = np.where(df[col] < lower_bound, Q1, df[col])
        df[col] = np.where(df[col] > upper_bound, Q3, df[col])
    
    return df

# Replace outliers in the dataset
data_cleaned = replace_outliers(data)

# Check the dataset after outlier replacement
print(data_cleaned.describe())

# Save the cleaned dataset with replaced outliers
data_cleaned.to_csv('cleaned_data.csv', index=False)













