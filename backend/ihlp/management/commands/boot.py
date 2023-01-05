from django.core.management.base import BaseCommand
from ihlp.management.boot import Boot


class Command(BaseCommand):
    def handle(self, **options):
        Boot().load()