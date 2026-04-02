import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Load processed data
df = pd.read_csv('data/processed/train.csv')

# Split features & target
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create models folder
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/model.pkl')

print("Model trained and saved")
