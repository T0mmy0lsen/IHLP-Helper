from django.urls import path

from . import views

urlpatterns = [
    path('request', views.request, name='request'),
    path('message', views.message, name='message'),
]