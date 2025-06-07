#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# Create models folder
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    """Read Yellow taxi trip data and prepare it."""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    
    # Print the number of records loaded (Question 3)
    print(f"Number of records loaded: {len(df)}")
    
    # Calculate duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter durations between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical columns to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Print the size after preparation (Question 4)
    print(f"Size of the result after preparation: {len(df)}")
    
    return df

def create_X(df, dv=None):
    """Create feature matrix using DictVectorizer."""
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    """Train a linear regression model and log it with MLflow."""
    with mlflow.start_run() as run:
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Print the intercept (Question 5)
        print(f"Intercept of the model: {model.intercept_}")
        
        # Calculate RMSE on validation set
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)
        
        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
        
        # Log the DictVectorizer
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        return run.info.run_id

def run(year, month):
    """Main function to run the pipeline."""
    # Load training and validation data
    df_train = read_dataframe(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)
    
    # Prepare features
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    # Target variable
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    # Train and log the model
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    
    run_id = run(year=args.year, month=args.month)
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)