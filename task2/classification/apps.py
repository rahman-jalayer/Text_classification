from django.apps import AppConfig

from classification.models import Prediction
import os
import pandas as pd


class ClassificationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classification'
    predictor = Prediction()
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset", "test.csv"), engine='python',
                            encoding='utf-8')
    test_data["text"] = test_data["Title"] + '. ' + test_data["Snippet"]
    test_data.set_index("URL")
    test_data.set_index("Title")
