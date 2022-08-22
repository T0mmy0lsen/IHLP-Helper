import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
# python manage.py runserver
from django.views.decorators.csrf import csrf_exempt

from app import settings

from schedule.parallel_machine_models import run

@csrf_exempt
def set_predict(request):
    text = json.loads(request.body)['text']
    return HttpResponse(run(text), content_type="application/json")

def get_predict(request):
    return HttpResponse(None, content_type="application/json")