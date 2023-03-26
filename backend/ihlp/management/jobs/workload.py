import pandas as pd

from ihlp.models import Workload, Responsible
from ihlp.models_ihlp import Request


def createWorkload(amount=0):

    queryset_requests = Request.objects.using('ihlp').order_by('-id')[:amount]

    df = pd.DataFrame.from_records(queryset_requests.values())
    df = df.fillna('')

    df_responsible = pd.DataFrame.from_records(
        Responsible.objects.filter(request_id__in=list(df.id.values)).values('id', 'data'))

    placement_arr = dict()
    responsible_arr = dict()

    for i, el in df_responsible.iterrows():

        placement = el.data['true_placement']
        responsible = el.data['true_responsible']

        placement_arr[placement] = placement_arr.get(placement, 0) + 1
        responsible_arr[responsible] = responsible_arr.get(responsible, 0) + 1

    Workload(data={
        'placement': placement_arr,
        'responsible': responsible_arr,
    }).save()