from datetime import datetime, timedelta
from django.db.models import Q
from ihlp.models_ihlp import Request

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

    return list(queryset_requests.values())

