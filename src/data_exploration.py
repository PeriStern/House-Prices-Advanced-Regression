import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def explore_data(data):
    """Display basic info and first few rows of the dataset."""
    print(data.info())
    print(data.head())
    print(f"Shape of the dataset: {data.shape}")

def display_unique_values(data):
    """Display unique value counts per column."""
    return data.nunique().to_frame().rename(columns={0: 'Unique Values'}).style.format({'Unique Values': '{:,.0f}'})

def plot_saleprice_distribution(data):
    """Plot the distribution of SalePrice."""
    plt.hist(data['SalePrice'])
    plt.title("SalePrice Distribution")
    plt.show()

def plot_feature_correlations(data):
    """Explore the correlation of each feature with SalePrice."""
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(14, len(numerical_columns) * 3))
    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(len(numerical_columns), 2, idx)
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(f"{feature} vs SalePrice")

    plt.tight_layout()
    plt.show()

def plot_feature_skewness(data):
    """Explore skewness in numerical features."""
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(14, len(numerical_columns) * 3))
    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(len(numerical_columns), 2, idx)
        sns.histplot(data[feature], kde=True)
        plt.title(f"{feature} | Skewness: {round(data[feature].skew(), 2)}")

    plt.tight_layout()
    plt.show()

def plot_non_numerical_features(data):
    """Explore relationships between categorical features and SalePrice."""
    non_numerical_columns = data.select_dtypes(include=['object']).columns

    plt.figure(figsize=(14, len(non_numerical_columns) * 3))
    for idx, feature in enumerate(non_numerical_columns, 1):
        plt.subplot(len(non_numerical_columns), 2, idx)
        sns.boxplot(x=data[feature], y=data['SalePrice'])
        plt.title(f"{feature} vs SalePrice")

    plt.tight_layout()
    plt.show()

def explore_category_cardinality(data):
    """Explore the cardinality of categorical features."""
    non_numerical_columns = data.select_dtypes(include=['object']).nunique()
    return non_numerical_columns[non_numerical_columns > 10]

if __name__ == "__main__":
    # Example usage
    train_data = load_data('../data/raw/train.csv')

    explore_data(train_data)
    print(display_unique_values(train_data))
    plot_saleprice_distribution(train_data)
    plot_feature_correlations(train_data)
    plot_feature_skewness(train_data)
    plot_non_numerical_features(train_data)
    print(explore_category_cardinality(train_data))
