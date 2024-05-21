from rest_framework import serializers
from .models import Answer


class RestApiSerializer(serializers.ModelSerializer):
    class Meta:
        model = Answer
        fields = ('query', 'answer')
