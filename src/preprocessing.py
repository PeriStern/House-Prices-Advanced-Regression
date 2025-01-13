import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from the given file path."""
    return pd.read_csv(file_path)

def clean_data(train):
    """Clean the dataset by handling missing values and normalizing features."""
    # Handle basement-related columns
    train.drop([948, 332], inplace=True)

    # Replace missing LotFrontage values with the median
    median_lot_frontage = train['LotFrontage'].median()
    train['LotFrontage'] = train['LotFrontage'].fillna(median_lot_frontage).astype(int)

    # Drop problematic rows
    train.drop([1379], inplace=True)

    # Replace remaining missing values with 'None'
    train.fillna('None', inplace=True)

    # Normalize the SalePrice column to minimize outlier effects
    train['SalePrice'] = np.log(train['SalePrice'])

    return train

def split_data(train, processed_train_path, processed_test_path):
    """Split the data into train and test sets and save to specified paths."""
    train_data, test_data = train_test_split(
        train, random_state=104, test_size=0.2, shuffle=True
    )
    train_data.to_csv(processed_train_path, index=False)
    test_data.to_csv(processed_test_path, index=False)
    print("Data preprocessing complete.")
    print(f"Train data saved to {processed_train_path}")
    print(f"Test data saved to {processed_test_path}")