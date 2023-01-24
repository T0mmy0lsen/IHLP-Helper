import pandas as pd

from datetime import datetime
from django.db.models import Q

from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload
from ihlp.models import Workload, WorkloadTotal
from ihlp.models_ihlp import Request


def createWorkloadTotal(
        from_date=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        hard_limit=100,
        predict=False
):
    queryset_requests = Request.objects.using('ihlp').filter(
        (Q(receiveddate__gte=from_date) | Q(receiveddate=None)) & Q(closingcode='0') & Q(solutiondate__isnull=True)).order_by('-id')[:hard_limit]

    df = pd.DataFrame.from_records(queryset_requests.values())
    df = df.fillna('')

    if predict:
        calculatePrediction(df=df)
        calculateWorkload(df=df)

    df_workloads = pd.DataFrame.from_records(
        Workload.objects.filter(request_id__in=list(df.id.values)).values('id', 'data'))

    responsibles = dict()
    placements = dict()

    for i, el in df_workloads.iterrows():

        placement = el.data['true_placement']
        responsible = el.data['true_responsible']

        if responsible in responsibles:
            responsibles[responsible] += 1
        else:
            responsibles[responsible] = 1

        if placement in placements:
            placements[placement] += 1
        else:
            placements[placement] = 1

    WorkloadTotal(data={
        'responsible': responsibles,
        'placement': placements,
    }).save()