from django.forms import model_to_dict

from ihlp.models import Workload, Predict


def setPredict(_):
    return False

def getPredict(data):
    for el in data:
        el['predict'] = model_to_dict(Predict.objects.filter(request_id=el['id']).first())
        el['workload'] = model_to_dict(Workload.objects.filter(request_id=el['id']).first())
    return data