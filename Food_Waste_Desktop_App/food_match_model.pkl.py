import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset again
df = pd.read_csv("food_waste.csv")

# Preprocess your data (Modify based on your original preprocessing steps)
X = df.drop(columns=["target_column"])  # Replace with actual target column
y = df["target_column"]

# Train the model again
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "food_waste_model.pkl")
print("Model saved successfully as food_waste_model.pkl")
