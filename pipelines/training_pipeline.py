from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_df
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.evaluation import evaluation

from steps.config import ModelNameConfig

# docker_settings = DockerSettings(required_integrations=[MLFLOW])


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
@pipeline(enable_cache=True)
def train_pipeline(data_path):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(
        x_train,
        x_test,
        y_train,
        y_test,
        config=ModelNameConfig(model_name="LinearRegression", fine_tuning=True),
    )
    mse, rmse = evaluation(model, x_test, y_test)
