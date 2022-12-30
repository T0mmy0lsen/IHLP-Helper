
import pandas as pd

from datetime import datetime, timedelta
from django.db.models import Q
from ihlp.models_ihlp import Request


def getRequest(text):
    queryset_requests = Request.objects.using('ihlp').filter(id=int(text))
    return pd.DataFrame.from_records(queryset_requests.values())

def getRequestLike(text, time=datetime.now(), limit=1):

    latest = time - timedelta(days=limit)

    if text == "":
        queryset_requests = Request.objects.using('ihlp').filter(
            Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
        )
    else:
        queryset_requests = Request.objects.using('ihlp').filter(
            (Q(subject__contains=text) | Q(description__contains=text)) &
            (Q(receiveddate__lte=time) & Q(receiveddate__gte=latest))
        )

    return pd.DataFrame.from_records(queryset_requests.values())

def getRequestLatest(text):

    if text == "":
        queryset_requests = Request.objects.using('ihlp').order_by('-id')[:10]
    else:
        queryset_requests = Request.objects.using('ihlp').filter(
            Q(subject__contains=text) | Q(description__contains=text)
        ).order_by('-id')[:10]

    return pd.DataFrame.from_records(queryset_requests.values())

