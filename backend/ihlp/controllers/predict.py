from django.forms import model_to_dict

from ihlp.models import Workload, Predict


def setPredict(_):
    return False


def getPredict(df):
    df['predict'] = df.apply(lambda x: model_to_dict(Predict.objects.filter(request_id=x['id']).first()), axis=1)
    df['workload'] = df.apply(lambda x: model_to_dict(Workload.objects.filter(request_id=x['id']).first()), axis=1)
    return df