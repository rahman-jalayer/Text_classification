from abc import ABC

from rest_framework import serializers
from django.core.validators import RegexValidator


class ClassificationSerializer(serializers.Serializer):
    title = serializers.CharField(required=True, max_length=500)
    url = serializers.CharField(required=True, max_length=500)

    class Meta:
        fields = ['title', 'url']


class ClassificationTextInputSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, max_length=5000)

    class Meta:
        fields = ['text']
