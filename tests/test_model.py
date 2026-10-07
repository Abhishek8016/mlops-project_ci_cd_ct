import os
import joblib

def test_model_exists_or_can_train():
    if not os.path.exists("models/model.pkl"):
        os.system("python src/train.py")
    
    assert os.path.exists("models/model.pkl")
    model = joblib.load("models/model.pkl")
    assert model is not None
