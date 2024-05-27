from django.db import models


# Create your models here.
class LlamaCpp(models.Model):
    query = models.TextField()
    answer = models.TextField()
    chatLog = models.TextField(default=None, null=True)
    
class Answer(models.Model):
    query = models.TextField()
    answer = models.TextField()