from mlProject.config.configuration import ModelEvaluationConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from pathlib import Path
from mlProject.utils.common import save_json



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        # saving metrics as local
        scores = {'rmse': rmse, 'mae': mae, 'r2': r2} 
        save_json(path= Path(self.config.metric_file_name), data=scores)


        # with mlflow.start_run():
        #     predicted_qualities = model.predict(test_x)

        #     (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        #     # saving metrics as local
        #     scores = {'rmse': rmse, 'mae': mae, 'r2': r2} 
        #     save_json(path= Path(self.config.metric_file_name), data=scores)

        #     mlflow.log_params(self.config.all_params)

        #     mlflow.log_metric('rmse', rmse)
        #     mlflow.log_metric('mae', mae)
        #     mlflow.log_metric('r2',r2)

        #     # log metrics does not work with file store
        #     if tracking_uri_type_store !='file':
        #         mlflow.sklearn.log_model(model, 'model', registered_model_name='ElasticnetModel')

        #     else:
        #         mlflow.sklearn.load_model(model, 'model')
