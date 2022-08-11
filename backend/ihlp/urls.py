from django.urls import path

from . import views

urlpatterns = [
    path('search/<int:request_id>/', views.get_request, name='request'),
    path('result/<int:request_id>/', views.get_request_result, name='request_result'),
]