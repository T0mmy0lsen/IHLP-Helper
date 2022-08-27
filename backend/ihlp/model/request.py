from ihlp.models_ihlp import Request, Item
from django.forms import model_to_dict
import json


def to_json_list(lst):
    return json.dumps([model_to_dict(m) for m in lst], indent=4, sort_keys=True, default=str)


def to_json(obj):
    _dict = model_to_dict(obj)
    _json = json.dumps(_dict, indent=4, sort_keys=True, default=str)
    return _json


def getRequestLike(text=None):
    request = Request.objects.filter(description__contains=text)[:10].using('ihlp')
    return to_json_list(request)


def getItem(id=None):
    item = Item.objects.filter(id=id).using('ihlp').first()
    return to_json(item)

