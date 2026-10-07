import os

# 1. Define folder structure
folders = [
    ".github/workflows",
    "data",
    "src",
    "tests",
    "models",
]

for folder in folders:
  os.makedirs(folder, exist_ok=True)
  print(f"Created directory: {folder}")

# 2. Write requirements.txt
requirements = """scikit-learn>=1.5.0
pandas>=2.2.0
pytest>=8.0.0
dvc>=3.48.0
mlflow>=2.11.0
joblib>=1.3.2
"""
with open("requirements.txt", "w") as f:
  f.write(requirements)
print("Created requirements.txt")

# 3. Write sample dataset data/raw_data.csv
csv_data = """feature1,feature2,target
0.5,1.2,0
1.5,0.8,1
0.1,0.2,0
1.8,1.9,1
0.4,0.9,0
1.2,1.5,1
0.3,0.1,0
1.9,1.7,1
"""
with open("data/raw_data.csv", "w") as f:
  f.write(csv_data)
print("Created data/raw_data.csv")

# 4. Write training script src/train.py
train_code = """import os
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
"""
with open("src/train.py", "w") as f:
  f.write(train_code)
print("Created src/train.py")

# 5. Write unit test tests/test_model.py
test_code = """import os
import joblib

def test_model_exists_or_can_train():
    if not os.path.exists("models/model.pkl"):
        os.system("python src/train.py")
    
    assert os.path.exists("models/model.pkl")
    model = joblib.load("models/model.pkl")
    assert model is not None
"""
with open("tests/test_model.py", "w") as f:
  f.write(test_code)
print("Created tests/test_model.py")

# 6. Write CI workflow
ci_yml = """name: CI Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run Tests with Pytest
        run: pytest tests/
"""
with open(".github/workflows/ci.yml", "w") as f:
  f.write(ci_yml)

# 7. Write CT workflow
ct_yml = """name: CT Pipeline

on:
  schedule:
    - cron: '0 0 * * 0'
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Pull Data via DVC
        run: dvc pull || echo "No remote configured, using local data"
      - name: Retrain Model & Track via MLflow
        run: python src/train.py
      - name: Version Model with DVC
        run: |
          dvc add models/model.pkl
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add models/model.pkl.dvc .dvc/ignore
          git commit -m "Auto-retrained and versioned model via CT" || echo "No changes"
          git push
"""
with open(".github/workflows/ct.yml", "w") as f:
  f.write(ct_yml)

# 8. Write CD workflow
cd_yml = """name: CD Pipeline

on:
  push:
    branches: [ "main" ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Container Image
        run: docker build -t ml-app-inference .
      - name: Deploy Placeholder
        run: echo "Model container built successfully and ready for deployment!"
"""
with open(".github/workflows/cd.yml", "w") as f:
  f.write(cd_yml)

# 9. Write Dockerfile
dockerfile = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/train.py"]
"""
with open("Dockerfile", "w") as f:
  f.write(dockerfile)

print("All project files created successfully!")