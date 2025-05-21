import os
import pickle
import click
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train-v2")
    
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.autolog()
    
    # Load the data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # Start MLflow run
    with mlflow.start_run():

        # Define model
        rf = RandomForestRegressor(max_depth=10, random_state=0)

        # Log model hyperparameters
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 0)

        # Train
        rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"RMSE: {rmse:.2f}")

        # Log evaluation metric
        mlflow.log_metric("rmse", rmse)
        # Log model
        mlflow.sklearn.log_model(rf, "model")



if __name__ == '__main__':
    run_train()
