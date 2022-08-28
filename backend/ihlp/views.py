import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
# python manage.py runserver
from django.views.decorators.csrf import csrf_exempt

from app import settings
from ihlp.boot import Boot
from ihlp.model.request import getRequestLike
from schedule.parallel_machine_models import setPredict, getPredict


@csrf_exempt
def boot(request):
    boot = Boot()
    boot.create_responsible()
    return HttpResponse("{\"message\": \"booted.\"}", content_type="application/json")


@csrf_exempt
def what(request):
    return HttpResponse("{\"message\": \"what.\"}", content_type="application/json")


@csrf_exempt
def predict(request):

    if request.method == 'GET':
        id = request.GET['id']
        return HttpResponse(getPredict(id), content_type="application/json")

    if request.method == 'POST':
        id = json.loads(request.body)['params']['id']
        user = json.loads(request.body)['params']['user']
        time = json.loads(request.body)['params']['time']
        keep = json.loads(request.body)['params']['keep']
        return HttpResponse(setPredict(id, user, time, keep), content_type="application/json")

    return what(request)


@csrf_exempt
def request(request):

    if request.method == 'GET':
        text = request.GET['text']
        return HttpResponse(getRequestLike(text), content_type="application/json")

    return what(request)