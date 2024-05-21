from django.urls import include, path
from . import views

app_name = 'restApiTest'
urlpatterns = [
    path('llama/', views.llama, name='request-llama'),
    path('', views.prompt_test_page, name='prompt_test_page'),
    path('generator/<str:query>', views.prompt_generator, name='request_generator')
]