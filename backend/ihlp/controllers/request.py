
import pandas as pd

from datetime import datetime, timedelta
from django.db.models import Q
from ihlp.models_ihlp import Request


def getRequest(text, limit=None):

    queryset_requests = Request.objects.using('ihlp').filter(id=text)

    if limit is not None:
        queryset_requests = queryset_requests[:limit]

    return pd.DataFrame.from_records(queryset_requests.values())


def getRequestLike(text, limit=None):

    if text == "":
        queryset_requests = Request.objects.using('ihlp').order_by('-id')
    else:
        queryset_requests = Request.objects.using('ihlp').filter(
            (Q(subject__contains=text) | Q(description__contains=text))
        ).order_by('-id')

    if limit is not None:
        queryset_requests = queryset_requests[:limit]

    return pd.DataFrame.from_records(queryset_requests.values())




