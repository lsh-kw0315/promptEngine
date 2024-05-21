from django.urls import path
from . import views

app_name = 'restApi'
urlpatterns = [
    path('', views.prompt_page, name='prompt_page'),
    path('generator/<str:query>', views.prompt_generator, name='request_generator'),
    path('llama2/', views.llama2, name='request-llama2'),
    path('geval/',views.geval,name='request-geval')
]