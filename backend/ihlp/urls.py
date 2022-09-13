from django.urls import path

from . import views

urlpatterns = [
    path('boot', views.boot, name='boot'),
    path('reboot', views.reboot, name='reboot'),
    path('predict', views.predict, name='predict'),
    path('request', views.request, name='request'),
]