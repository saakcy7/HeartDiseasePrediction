import tkinter as tk
from tkinter import messagebox
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

root = tk.Tk()

root.title("Heart Disease Prediction System")
root.geometry("500x700")

title = tk.Label(
    root,
    text="Heart Disease Prediction",
    font=("Arial", 18, "bold")
)

title.pack(pady=20)

labels = [
    "Age",
    "Sex (1=Male, 0=Female)",
    "Chest Pain Type (0-3)",
    "Resting Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar (1=True, 0=False)",
    "Rest ECG (0-2)",
    "Maximum Heart Rate",
    "Exercise Induced Angina (1=Yes, 0=No)",
    "Oldpeak",
    "Slope (0-2)",
    "CA (0-4)",
    "Thal (1-3)"
]

entries = []

for label_text in labels:

    frame = tk.Frame(root)
    frame.pack(pady=5)

    label = tk.Label(frame, text=label_text, width=35, anchor="w")
    label.pack(side=tk.LEFT)

    entry = tk.Entry(frame, width=20)
    entry.pack(side=tk.RIGHT)

    entries.append(entry)

def predict():

    try:

        values = []

        for entry in entries:
            values.append(float(entry.get()))

        sample = [values]

        sample = scaler.transform(sample)

        prediction = model.predict(sample)

        if prediction[0] == 1:

            result_label.config(
                text="Heart Disease Detected",
                fg="red"
            )

        else:

            result_label.config(
                text="No Heart Disease",
                fg="green"
            )

    except:

        messagebox.showerror(
            "Error",
            "Please enter valid numeric values."
        )

predict_button = tk.Button(
    root,
    text="Predict",
    font=("Arial", 14, "bold"),
    bg="blue",
    fg="white",
    command=predict
)

predict_button.pack(pady=20)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=20)

root.mainloop()