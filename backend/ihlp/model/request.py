from django.db.models import Q

from helpers.converters import to_json_list, to_json
from ihlp.models_ihlp import Request, Item
from datetime import datetime

def getRequestLike(text, time):
    queryset_requests = Request.objects.using('ihlp').filter(
        (Q(subject__contains=text) | Q(description__contains=text)) &
        (Q(solutiondate__gte=time) | (Q(solutiondate=None) & Q(solution=None) & Q(closingcode=None))) &
        (Q(receiveddate__lte=time))
    )
    return list(queryset_requests.values())

