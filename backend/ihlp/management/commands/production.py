import os
import time

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")

from datetime import datetime
from django.core.management.base import BaseCommand
from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.responsibles import calculateResponsibles
from ihlp.management.jobs.workload import createWorkload


class Command(BaseCommand):
    def handle(self, **options):
        start_time = datetime.now()
        self.run()
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

    def run(self):

        calculatePrediction(
            amount=10,
            delete=True
        )

        calculateResponsibles(
            amount=100,
            delete=True
        )

        createWorkload(
            amount=100
        )




