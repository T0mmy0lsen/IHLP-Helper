from django.forms import model_to_dict

from ihlp.models import Workload, Predict


def setPredict(_):
    return False


def getPredict(df):

    def applyGetPredict(x):
        p = Predict.objects.filter(request_id=x['id']).first()
        if p is None:
            return False
        return model_to_dict(p)

    def applyGetWorkload(x):
        w = Workload.objects.filter(request_id=x['id']).first()
        if w is None:
            return False
        return model_to_dict(w)

    df['predict'] = df.apply(lambda x: applyGetPredict(x), axis=1)
    df['workload'] = df.apply(lambda x: applyGetWorkload(x), axis=1)

    return df