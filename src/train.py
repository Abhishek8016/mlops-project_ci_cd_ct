import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("CI_CD_CT_Pipeline")

def train():
    with mlflow.start_run():
        data_path = "data/raw_data.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}. Run 'dvc pull' first.")
        
        df = pd.read_csv(data_path)
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_estimators = 50
        max_depth = 3
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print(f"Training Successful! Accuracy: {acc:.2f}, F1-Score: {f1:.2f}")

if __name__ == "__main__":
    train()
