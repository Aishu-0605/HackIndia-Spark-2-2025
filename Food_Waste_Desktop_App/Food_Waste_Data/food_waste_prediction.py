import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Load the trained model
with open("food_waste_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to predict food waste category
def predict_waste():
    try:
        quantity = float(entry_quantity.get())
        storage_time = float(entry_storage.get())
        food_type = float(entry_food_type.get())

        # Create input array
        input_features = np.array([[quantity, storage_time, food_type]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]

        # Map prediction to category names
        categories = {0: "Low Waste", 1: "Moderate Waste", 2: "High Waste"}
        category_name = categories.get(prediction, "Unknown")

        # Show result in a message box
        messagebox.showinfo("Prediction", f"Food Waste Category: {category_name}")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values!")

# Create the GUI window
root = tk.Tk()
root.title("Food Waste Prediction")

# Labels & Entry fields
tk.Label(root, text="Quantity of Food:").grid(row=0, column=0)
entry_quantity = tk.Entry(root)
entry_quantity.grid(row=0, column=1)

tk.Label(root, text="Storage Time (days):").grid(row=1, column=0)
entry_storage = tk.Entry(root)
entry_storage.grid(row=1, column=1)

tk.Label(root, text="Food Type / Temperature:").grid(row=2, column=0)
entry_food_type = tk.Entry(root)
entry_food_type.grid(row=2, column=1)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict_waste)
predict_button.grid(row=3, column=1)

# Run the GUI
root.mainloop()
