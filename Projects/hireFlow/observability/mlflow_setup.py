import mlflow
from config import settings


def setup_mlflow() -> str:
    """
    Initialize MLflow with SQLite backend.
    SQLite is the default in latest MLflow — no server needed for tracking.

    DB file: ./mlflow.db (auto-created in hireFlow/ root)
    View UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    """
    # SQLite backend — auto-creates mlflow.db in current directory
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=settings.mlflow_experiment_name,
            tags={
                "project": "HireFlow",
                "version": "1.0",
                "team": "RByte.ai",
            }
        )
        print(f"Created MLflow experiment: {settings.mlflow_experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using MLflow experiment: {settings.mlflow_experiment_name} (id: {experiment_id})")

    mlflow.set_experiment(settings.mlflow_experiment_name)
    return experiment_id