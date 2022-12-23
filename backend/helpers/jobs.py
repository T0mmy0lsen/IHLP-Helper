from django.db.models import Q
from helpers.converters import to_json, to_json_list
from ihlp.models_ihlp import Request, Relation, Item, Object, RelationHistory, ObjectHistory
from datetime import datetime, timedelta


def calculateWorkload(now=datetime.now()):

    latest = now - timedelta(days=1)

    queryset_requests = Request.objects.using('ihlp').filter(Q(receiveddate__gte=latest) & (Q(solutiondate__gte=now) | Q(solutiondate=None)))
    object_requests = list(queryset_requests.values('id'))
    list_of_request__id = [v['id'] for v in object_requests]

    queryset_relations = RelationHistory.objects.using('ihlp').filter(leftid__in=list_of_request__id)
    object_relations = list(queryset_relations.values('rightid', 'tblid'))
    list_of_object__id = [v['tblid'] for v in object_relations]
    list_of_item__id = [v['rightid'] for v in object_relations]

    queryset_objects = ObjectHistory.objects.using('ihlp').filter(tblid__in=list_of_object__id)
    queryset_items = Item.objects.using('ihlp').filter(id__in=list_of_item__id)

    object_objects = list(queryset_objects.values())
    object_items = list(queryset_items.values())
    return [object_requests, object_relations, object_objects, object_items]