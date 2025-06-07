from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Define the DAG with default params
with DAG(
    dag_id='nyc_taxi_training',
    default_args=default_args,
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False,
    params={
        'year': 2023,  # Default year
        'month': 3     # Default month
    }
) as dag:
    # Task to run the training script
    run_training = BashOperator(
        task_id='run_training',
        bash_command='python /opt/airflow/dags/train.py --year {{ dag_run.conf["year"] if dag_run.conf else 2023 }} --month {{ dag_run.conf["month"] if dag_run.conf else 3 }}',
    )

