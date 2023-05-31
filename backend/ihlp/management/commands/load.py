from django.core.management.base import BaseCommand
from ihlp.management.load import Load


class Command(BaseCommand):
    def handle(self, **options):
        Load().load()