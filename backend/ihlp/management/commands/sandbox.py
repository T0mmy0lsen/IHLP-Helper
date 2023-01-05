
from datetime import datetime
from django.core.management.base import BaseCommand

from ihlp.management.jobs.evaluate import evaluateWorkloadWithUserPredictionAndSchedule, evaluate
from django.core.management.base import BaseCommand

from ihlp.management.boot import Boot
from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload



class Command(BaseCommand):
    def handle(self, **options):

        time = datetime.strptime("2022-03-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        limit = 28
        df = None

        # evaluateWorkloadWithUserPredictionAndSchedule(time=time, limit=limit, df=df)
        evaluate()
