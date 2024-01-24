import pandas as pd
from mlProject.config.configuration import ModelTrainerConfig
import os
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score



class ModelTrainer:
    def __init__(self,
                 config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predict = lr.predict(test_x)

        r2_value = r2_score(test_y, predict)
        print(f"r2_value is : {r2_value}")

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))