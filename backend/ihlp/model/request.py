from helpers.converters import to_json_list, to_json
from ihlp.models_ihlp import Request, Item
from datetime import datetime

def getRequestLike(text=None, time=datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
    request = Request.objects.filter(description__contains=text)[:10].using('ihlp')
    return to_json_list(request)

