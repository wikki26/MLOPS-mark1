from src.predict import predict_failure


def test_prediction_output():
    result = predict_failure(
        temperature=85,
        vibration=0.75,
        pressure=42,
        runtime_hours=220
    )

    assert result in ["Failure Likely", "Normal"]