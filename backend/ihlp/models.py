from django.db import models
from jsonfield import JSONField

# > python manage.py makemigrations ihlp
# > python manage.py migrate

class Predict(models.Model):

    request_id = models.IntegerField()
    data = JSONField()

    def __str__(self):
        return "Predict: " + str(self.request_id)

class Workload(models.Model):

    request_id = models.IntegerField()
    username = models.CharField(max_length=100, blank=True, null=True)
    data = JSONField()

    def __str__(self):
        return "Predict: " + str(self.request_id)




