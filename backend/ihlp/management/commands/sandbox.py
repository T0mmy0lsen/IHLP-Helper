from django.core.management.base import BaseCommand

from ihlp.management.boot import Boot
from ihlp.management.jobs.prediction import calculatePrediction
from ihlp.management.jobs.workload import calculateWorkload


class Command(BaseCommand):
    def handle(self, **options):
        # Boot().load()
        calculatePrediction()
        calculateWorkload()