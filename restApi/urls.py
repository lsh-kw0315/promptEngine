from django.urls import path
from . import views

app_name = 'restApi'
urlpatterns = [
    path('', views.prompt_page, name='prompt_page'),
    path('hautogenerator/<str:query>', views.halfauto_generator, name='request_hautogenerator'),
    path('autogenerator/<str:query>', views.auto_generator, name='request_autogenerator'),
    path('geval/',views.geval,name='request-geval'),
    path('analysis/', views.analysis, name='request-analysis'),
    path('improve/', views.improve, name='improve-analysis'),
]