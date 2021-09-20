from django.db import models
from task_manager.models import Step
from library.models import Paper


class Normalization(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="norm_dataframe")
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="norm_model")
    transformed = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="norm_transformed")


class Column(models.Model):
    algorithm = models.ForeignKey(Normalization, models.CASCADE)
    name = models.TextField()
    x = models.BooleanField(default=False)

    def __str__(self):
        return self.name
