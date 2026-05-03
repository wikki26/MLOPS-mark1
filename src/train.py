from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "machine_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def train_model():
    data = pd.read_csv(DATA_PATH)

    features = data[["temperature", "vibration", "pressure", "runtime_hours"]]
    target = data["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Training completed.")
    print(f"Accuracy: {accuracy}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()