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
        return "Workload: " + str(self.request_id)


class WorkloadTotal(models.Model):

    id = models.AutoField(primary_key=True)
    data = JSONField()

    def __str__(self):
        return "WorkloadTotal: " + str(self.id)


class Feedback(models.Model):

    request_id = models.IntegerField()
    type = models.CharField(max_length=100, blank=True, null=True)
    message = models.CharField(max_length=2000, blank=True, null=True)

    def __str__(self):
        return "Feedback: " + str(self.request_id)



