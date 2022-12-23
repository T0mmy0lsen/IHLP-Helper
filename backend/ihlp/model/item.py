from helpers.converters import to_json
from ihlp.models_ihlp import Item


def getItem(id=None):
    item = Item.objects.filter(id=id).using('ihlp').first()
    return to_json(item)