from rest_framework import serializers
from .models import LlamaCpp


class RestApiSerializer(serializers.ModelSerializer):
    class Meta:
        model = LlamaCpp
        fields = ('query', 'answer')
