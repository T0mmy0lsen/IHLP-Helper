from datetime import datetime
from django.core.management.base import BaseCommand

from ihlp.management.jobs.evaluate import evaluateWorkloadWithUserPredictionAndSchedule, evaluate


class Command(BaseCommand):
    def handle(self, **options):
        time = datetime.strptime("2022-03-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        limit = 28
        df = None

        # evaluateWorkloadWithUserPredictionAndSchedule(time=time, limit=limit, df=df)
        evaluate()
