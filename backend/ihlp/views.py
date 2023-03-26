import json
import pandas as pd

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
# python manage.py runserver

from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from ihlp.controllers.request import getRequestLike, getRequest
from ihlp.controllers.predict import getPredict

from ms_identity_web.django.middleware import ms_identity_web
from ihlp.models import Feedback, Predict, Workload
from ihlp.models_ihlp import Request

from django.conf import settings

if settings.SECURE:
    @ms_identity_web.login_required
    def index(request):
        return render(request, 'index.html')
else:
    def index(request):
        return render(request, 'index.html')


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

        predict = predict == 'true'

        if predict:
            queryset_predicts = Predict.objects.order_by("-id")
            if limit is not None:
                queryset_predicts = queryset_predicts[:limit]
            ids = [e['request_id'] for e in list(queryset_predicts.values())]
            df = pd.DataFrame.from_records(Request.objects.using('ihlp').filter(id__in=ids).values())
        else:
            if text.isdigit():
                df = getRequest(int(text), limit=limit)
            else:
                df = getRequestLike(text, limit=limit)

        df = df.sort_values("id", ascending=False)

        if len(df) == 0:
            return JsonResponse({
                'text': text,
                'length': 0,
                'data': False
            }, safe=False)

        df = getPredict(df)

        result = df.to_json(orient="records")
        parsed = json.loads(result)

        return JsonResponse({
            'text': text,
            'length': len(df),
            'data': parsed,
            'workload': Workload.objects.order_by('-id')[0].data
        }, safe=False)

    return what(request)