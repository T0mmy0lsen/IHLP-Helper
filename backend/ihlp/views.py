import json

from django.http import HttpResponse, JsonResponse

# Create your views here.
# python manage.py runserver
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from ihlp.controllers.request import getRequestLike, getRequest
from ihlp.controllers.predict import getPredict
from datetime import datetime

from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload
from ihlp.models import Feedback, WorkloadTotal


@csrf_exempt
def what(request):
    return HttpResponse("{\"message\": \"what.\"}", content_type="application/json")


@csrf_exempt
@api_view(['GET'])
def message(request):

    if request.method == 'GET':

        id = int(request.GET.get('id', False))
        type = request.GET.get('type', '')
        message = request.GET.get('message', '')

        Feedback(
            request_id=id,
            type=type,
            message=message,
        ).save()

    return what(request)


@csrf_exempt
@api_view(['GET'])
def request(request):

    if request.method == 'GET':

        predict = request.GET.get('predict', False)
        limit = int(request.GET.get('limit', 0))
        text = request.GET.get('text', False)

        if text.isdigit():
            df = getRequest(int(text), limit=limit)
        else:
            df = getRequestLike(text, limit=limit)

        if len(df) == 0:
            return JsonResponse({
                'text': text,
                'length': 0,
                'data': False
            }, safe=False)

        if predict:
            calculatePrediction(df=df)
            calculateWorkload(df=df)

        df = getPredict(df)

        result = df.to_json(orient="records")
        parsed = json.loads(result)

        return JsonResponse({
            'text': text,
            'length': len(df),
            'data': parsed,
            'workload': WorkloadTotal.objects.order_by('-id')[0].data
        }, safe=False)

    return what(request)