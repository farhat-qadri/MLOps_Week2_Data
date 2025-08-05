import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
import sys

# <-- CHANGE: Pointing directly to your iris_v2.csv file
VALIDATION_DATA_PATH = "data/iris_v2.csv"
ACCURACY_THRESHOLD = 0.90 

def validate_model():
    print("Starting validation...")
    mlflow.set_tracking_uri("file:./mlruns")
    runs = mlflow.search_runs(experiment_ids="0", order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print("ERROR: No MLflow runs found. Exiting.")
        sys.exit(1)
    latest_run_id = runs.iloc[0].run_id
    print(f"Validating model from run ID: {latest_run_id}")
    try:
        logged_model_uri = f"runs:/{latest_run_id}/iris-model"
        model = mlflow.pyfunc.load_model(logged_model_uri)
    except Exception as e:
        print(f"ERROR: Could not load model from MLflow. Details: {e}")
        sys.exit(1)
    validation_data = pd.read_csv(VALIDATION_DATA_PATH)
    X_val = validation_data.drop("species", axis=1)
    y_val = validation_data["species"]
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"Model accuracy on clean validation set: {accuracy:.4f}")
    print(f"Required accuracy threshold: {ACCURACY_THRESHOLD:.4f}")
    if accuracy < ACCURACY_THRESHOLD:
        print("\n!!! VALIDATION FAILED: Model accuracy is below threshold. !!!")
        sys.exit(1)
    else:
        print("\n--- VALIDATION PASSED: Model performance is acceptable. ---")
        sys.exit(0)

if __name__ == "__main__":
    validate_model()
