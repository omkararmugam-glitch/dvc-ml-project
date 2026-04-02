import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load raw data
df = pd.read_csv('data/raw/iris.csv')

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create processed folder
os.makedirs('data/processed', exist_ok=True)

# Save files
train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print("Data preparation done")
