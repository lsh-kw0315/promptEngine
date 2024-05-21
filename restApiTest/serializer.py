from rest_framework import serializers
from .models import LlamaCpp


class RestApiTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = LlamaCpp
        fields = ('query', 'answer')
