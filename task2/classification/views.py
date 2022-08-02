from rest_framework.views import APIView
from django.http import JsonResponse

from classification.serializers import ClassificationSerializer, ClassificationTextInputSerializer
from rest_framework import status

from classification.apps import ClassificationConfig


class ClassificationListView(APIView):
    def get(self, request):
        serializer = ClassificationSerializer(data=request.query_params)
        if serializer.is_valid():
            url = serializer.data['url']
            title = serializer.data['title']
            data = ClassificationConfig.test_data
            df_text = data[(((data.URL == url) & (data.Title == title)))].reset_index(drop=True)
            text = ''
            if not df_text.empty:
                text = df_text.iloc[0]['text']
            else:
                return JsonResponse({'result': 'There are no any documents with these parameters'})
            result = ClassificationConfig.predictor.predict(text)
            return JsonResponse({'result': result})
        else:
            return JsonResponse({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)


class ClassificationTextInputListView(APIView):
    def get(self, request):
        serializer = ClassificationTextInputSerializer(data=request.query_params)
        if serializer.is_valid():
            text = serializer.data['text']
            result = ClassificationConfig.predictor.predict(text)
            return JsonResponse({'result': result})
        else:
            return JsonResponse({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
