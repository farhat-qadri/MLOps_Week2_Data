import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train(data_path, poison_level):
    print("Starting training run...")
    with mlflow.start_run():
        print(f"Logging parameters: data_path={data_path}, poison_level={poison_level}")
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("poison_level", poison_level)
        df = pd.read_csv(data_path)
        X = df.drop("species", axis=1)
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "iris-model")
        print("MLflow run completed.")
        print("To see results, run 'mlflow ui' and use the Web Preview feature in Cloud Shell.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model with MLflow.")
    parser.add_argument("--data-path", required=True, help="Path to the training data CSV.")
    parser.add_argument("--poison-level", type=int, default=0, help="The poison level percentage for logging.")
    args = parser.parse_args()
    train(args.data_path, args.poison_level)
