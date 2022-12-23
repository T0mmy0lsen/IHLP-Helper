import json

from django.http import HttpResponse, JsonResponse

# Create your views here.
# python manage.py runserver
from django.views.decorators.csrf import csrf_exempt

from helpers.boot import Boot
from helpers.jobs import calculateWorkload
from ihlp.model.request import getRequestLike
from ihlp.model.predict import setPredict, getPredict
from datetime import datetime

@csrf_exempt
def load(request):
    boot = Boot()
    boot.load()
    return HttpResponse("{\"message\": \"loaded.\"}", content_type="application/json")

@csrf_exempt
def boot(request):
    boot = Boot()
    boot.users()
    return HttpResponse("{\"message\": \"booted.\"}", content_type="application/json")

@csrf_exempt
def reboot(request):
    boot = Boot()
    boot.reset()
    boot.users()
    return HttpResponse("{\"message\": \"rebooted.\"}", content_type="application/json")


@csrf_exempt
def what(request):
    return HttpResponse("{\"message\": \"what.\"}", content_type="application/json")


@csrf_exempt
def predict(request):

    if request.method == 'GET':
        id = request.GET['id']
        return HttpResponse(getPredict(id), content_type="application/json")

    if request.method == 'POST':
        params = json.loads(request.body)['params']
        return HttpResponse(setPredict(params), content_type="application/json")

    return what(request)


@csrf_exempt
def request(request):

    if request.method == 'GET':
        text = request.GET['text']
        time = request.GET['time']
        return HttpResponse(getRequestLike(text, time), content_type="application/json")

    return what(request)


@csrf_exempt
def workload(request):

    if request.method == 'GET':
        return JsonResponse(calculateWorkload(now=datetime.strptime("2022-09-27 15:25:00", "%Y-%m-%d %H:%M:%S")), safe=False)

    return what(request)