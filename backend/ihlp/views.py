import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
# python manage.py runserver
from app import settings

from schedule.parallel_machine_models import run


def set_predict(request):
    return HttpResponse(run(), content_type="application/json")

def get_predict(request):
    return HttpResponse(None, content_type="application/json")