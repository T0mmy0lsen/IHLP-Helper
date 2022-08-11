import json

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
# python manage.py runserver
from backend.app import settings


def get_request(request, request_id):
    items = [
        {"id": 1, "predict": settings.svm_time.predict('Jeg skal have nulstillet min kode men jeg kan ikke logge ind med nem-id på password.sdu.dk.')},
        {"id": 2, "predict": settings.svm_responsible.predict('Jeg skal have nulstillet min kode men jeg kan ikke logge ind med nem-id på password.sdu.dk.')},
    ]
    return HttpResponse(json.dumps(items), content_type="application/json")


def get_request_result(request, request_id):
    items = [
        {"id": 1, "value": "Tommy Olsen"},
        {"id": 2, "value": "Tommy Petersen"},
        {"id": 3, "value": "Tommy Mortensen"},
    ]
    return HttpResponse(json.dumps(items), content_type="application/json")