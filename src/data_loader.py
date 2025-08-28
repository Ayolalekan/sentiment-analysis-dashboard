import pandas as pd

# Load the dataset
df = pd.read_csv('data/amazon_reviews.csv')

# Show first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

print("\nColumns in the dataset:")
print(df.columns)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDataset info:")
print(df.info())
