from classification.views import ClassificationListView, ClassificationTextInputListView
from django.urls import path

urlpatterns = [
    path("prediction/", ClassificationListView.as_view(), name='prediction'),
    path("prediction_text_input/", ClassificationTextInputListView.as_view(), name='prediction_text_input'),
]
