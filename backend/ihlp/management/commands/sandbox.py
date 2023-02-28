import os
import time

from ihlp.management.jobs.workload_total import createWorkloadTotal

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

from datetime import datetime
from django.core.management.base import BaseCommand
from ihlp.management.jobs.evaluate import evaluateWorkloadWithUserPredictionAndSchedule, evaluate
from django.core.management.base import BaseCommand
from ihlp.management.boot import Boot
from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload


class Command(BaseCommand):
    def handle(self, **options):
        start_time = datetime.now()
        self.run()
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

    def run(self):

        time = datetime.strptime("2022-03-01 10:00:00", "%Y-%m-%d %H:%M:%S")

        calculatePrediction(
            time=time,
            limit_days=0,
            limit_minutes=0,
            limit_amount=5,
            should_delete=False
        )

        calculateWorkload(
            time=time,
            limit_days=0,
            limit_minutes=0,
            limit_amount=10,
            should_delete=True
        )

        createWorkloadTotal(
            time=time,
            limit_days=0,
            limit_minutes=0,
            limit_amount=10
        )





