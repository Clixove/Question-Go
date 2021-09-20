from django.db import models
from task_manager.models import Step
from library.models import Paper


class PreProcessing(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    report = models.TextField(blank=True)

    def dataframe(self): return self.step.predicted_data


class Column(models.Model):
    algorithm = models.ForeignKey(PreProcessing, models.CASCADE)
    name = models.TextField()

    def __str__(self):
        return self.name
