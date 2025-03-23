pip install tensorflow flask pandas numpy scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    # Sample dataset (Replace with real medical dataset)
    data = {
        "fever": [1, 0, 1, 0, 1, 1, 0],
        "cough": [1, 1, 0, 1, 0, 1, 0],
        "fatigue": [0, 1, 1, 0, 1, 0, 1],
        "diagnosis": ["Flu", "COVID-19", "Cold", "Allergy", "Flu", "COVID-19", "Cold"]
    }

    df = pd.DataFrame(data)

    # Encode the diagnosis column
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    # Splitting features and labels
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return train_test_split(X, y, test_size=0.2, random_state=42), le
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from medical_data import load_data

# Load preprocessed data
(X_train, X_test, y_train, y_test), label_encoder = load_data()

# Define model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(len(set(y_train)), activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))

# Save the model
model.save("medical_diagnosis_model.h5")
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from medical_data import load_data

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("medical_diagnosis_model.h5")

# Load label encoder
(_, _, _, _), label_encoder = load_data()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = pd.DataFrame([data])  # Convert to DataFrame

        # Ensure input format
        if not all(col in symptoms.columns for col in ["fever", "cough", "fatigue"]):
            return jsonify({"error": "Invalid input format"}), 400

        prediction = model.predict(symptoms)
        diagnosis_index = np.argmax(prediction)
        diagnosis = label_encoder.inverse_transform([diagnosis_index])[0]

        return jsonify({"diagnosis": diagnosis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Medical Diagnosis</title>
</head>
<body>
    <h2>Enter Symptoms</h2>
    <form id="symptom-form">
        <label>Fever (1 for Yes, 0 for No):</label>
        <input type="number" id="fever" required><br>

        <label>Cough (1 for Yes, 0 for No):</label>
        <input type="number" id="cough" required><br>

        <label>Fatigue (1 for Yes, 0 for No):</label>
        <input type="number" id="fatigue" required><br>

        <button type="submit">Get Diagnosis</button>
    </form>

    <h3>Diagnosis: <span id="diagnosis-result"></span></h3>

    <script>
        document.getElementById("symptom-form").addEventListener("submit", function(event) {
            event.preventDefault();
            const fever = document.getElementById("fever").value;
            const cough = document.getElementById("cough").value;
            const fatigue = document.getElementById("fatigue").value;

            fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fever: Number(fever), cough: Number(cough), fatigue: Number(fatigue) })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("diagnosis-result").textContent = data.diagnosis;
            })
            .catch(error => console.error("Error:", error));
        });
    </script>
</body>
</html>
python app.py
python train_model.py
# Implementation-of-AI-Powered-Medical-Diagnosis-System
