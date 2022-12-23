import json
from collections.abc import Iterable

from django.forms import model_to_dict

def to_json_list(lst):
    return json.dumps([model_to_dict(m) for m in lst], indent=4, sort_keys=True, default=str)

def to_json(obj):
    if isinstance(obj, Iterable):
        return to_json_list(obj)
    _dict = model_to_dict(obj)
    _json = json.dumps(_dict, indent=4, sort_keys=True, default=str)
    return _json