import json

from django.http import HttpResponse, JsonResponse

# Create your views here.
# python manage.py runserver
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from ihlp.controllers.request import getRequestLike
from ihlp.controllers.predict import getPredict
from datetime import datetime

from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload


@csrf_exempt
def what(request):
    return HttpResponse("{\"message\": \"what.\"}", content_type="application/json")


@csrf_exempt
@api_view(['GET'])
def request(request):

    if request.method == 'GET':

        predict = request.GET.get('predict', False)
        text = request.GET.get('text', False)
        time = request.GET.get('time', False)

        # time = "2022-02-01 00:00:00"

        if not time or time == 'false':
            time = datetime.now()
        else:
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

        df = getRequestLike(text, time)

        if predict:
            calculatePrediction(df=df)
            calculateWorkload(df=df)

        df = getPredict(df)

        result = df.to_json(orient="records")
        parsed = json.loads(result)

        return JsonResponse({
            'text': text,
            'time': time,
            'length': len(df),
            'data': parsed
        }, safe=False)

    return what(request)