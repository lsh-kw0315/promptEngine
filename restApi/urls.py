from django.urls import path
from . import views

app_name = 'restApi'
urlpatterns = [
    path('', views.prompt_page, name='prompt_page'),
    path('generator/<str:query>', views.prompt_generator, name='request_generator'),
]