from django.db import models
from jsonfield import JSONField


class Predict(models.Model):

    time = models.IntegerField()
    responsible = JSONField()

    def __str__(self):
        return "Predict: " + str(self.id)


class Slot(models.Model):

    keep = models.BooleanField()
    user = models.CharField(max_length=255)
    time = models.IntegerField()
    index = models.IntegerField()

    def __str__(self):
        return "Slot: " + str(self.id)


class Machine(models.Model):

    user = models.CharField(max_length=255)
    slots = models.ManyToManyField(Slot, blank=True)

    def __str__(self):
        return "Machine: " + str(self.user)


class Schedule(models.Model):

    name = models.CharField(max_length=255)
    machines = models.ManyToManyField(Machine, blank=True)
    predicts = models.ManyToManyField(Predict, blank=True)

    def __str__(self):
        return "Schedule: " + self.name







