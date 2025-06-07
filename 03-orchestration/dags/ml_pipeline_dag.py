from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def load_data():
    print("Loading data...")

def preprocess_data():
    print("Preprocessing data...")

def train_model():
    print("Training model...")

def evaluate_model():
    print("Evaluating model...")

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval="*/1 * * * *",  # Run every minute using cron expression
    catchup=False,
    tags=["ml", "beginner"]
) as dag:

    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    t1 >> t2 >> t3 >> t4  # Define execution order