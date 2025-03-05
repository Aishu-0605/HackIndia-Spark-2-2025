import pickle
import numpy as np

# Load the trained model
with open("food_match_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example test data (Modify based on your dataset structure)
new_data = np.array([[2.5, 30, 1]])  # Example input values

# Make prediction
prediction = model.predict(new_data)
print(f"Prediction: {prediction}")
