import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("food_waste_model.pkl")

def predict_waste():
    try:
        # Get user inputs
        input_values = [
            float(entry_1.get()),
            float(entry_2.get()),
            float(entry_3.get()),
        ]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_values])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Show result
        messagebox.showinfo("Prediction", f"Food Waste Category: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Create GUI window
root = tk.Tk()
root.title("Food Waste Prediction")
root.geometry("400x350")

tk.Label(root, text="Feature 1:").pack()
entry_1 = tk.Entry(root)
entry_1.pack()

tk.Label(root, text="Feature 2:").pack()
entry_2 = tk.Entry(root)
entry_2.pack()

tk.Label(root, text="Feature 3:").pack()
entry_3 = tk.Entry(root)
entry_3.pack()

tk.Button(root, text="Predict", command=predict_waste).pack()

# Run the GUI
root.mainloop()
