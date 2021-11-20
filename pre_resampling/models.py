from django.db import models
from library.models import Paper
from task_manager.models import Step


class Resampling(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="resampling_dataframe")
    class_dict = models.TextField(blank=True)


class Column(models.Model):
    algorithm = models.ForeignKey(Resampling, models.CASCADE)
    name = models.TextField()
    y_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
