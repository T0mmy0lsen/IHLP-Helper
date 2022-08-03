from django.urls import path

from . import views

urlpatterns = [
    path('<int:request_id>/', views.get_request, name='request'),
]