from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def predict_failure(temperature, vibration, pressure, runtime_hours):
    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([{
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "runtime_hours": runtime_hours
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Failure Likely"
    else:
        return "Normal"


if __name__ == "__main__":
    result = predict_failure(
        temperature=85,
        vibration=0.75,
        pressure=42,
        runtime_hours=220
    )

    print(f"Prediction result: {result}")