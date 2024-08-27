from django.db import models


# Create your models here.
class Answer(models.Model):
    query = models.TextField()
    answer = models.TextField()
