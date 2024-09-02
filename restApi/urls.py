from django.urls import path
from . import views

app_name = 'restApi'
urlpatterns = [
    path('', views.prompt_page, name='prompt_page'),
    path('hautogenerator/<str:query>', views.gemini_prompt_halfauto_generator, name='request_hautogenerator'),
    path('autogenerator/<str:query>', views.gemini_prompt_auto_generator, name='request_autogenerator'),
    path('llama2/', views.llama2, name='request-llama2'),
    path('geval/',views.geval,name='request-geval'),
    path('analysis/', views.analysis, name='request-analysis')
]