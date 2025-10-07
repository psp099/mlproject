import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException("Error occurred while saving object", e)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
        return report
    except Exception as e:
        raise CustomException("Error occurred while evaluating models", e)