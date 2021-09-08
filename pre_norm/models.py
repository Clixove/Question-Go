from django.db import models
from task_manager.models import Step
from library.models import Paper


class Normalization(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="normalization_dataframe")
    note = models.TextField(blank=True)
    error_message = models.TextField(blank=True)

    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="norm_model")
    predict = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="norm_predict")

    def open_permission(self, user):
        return any([x.user == user for x in self.step.task.openedtask_set.all()])


class Column(models.Model):
    algorithm = models.ForeignKey(Normalization, models.CASCADE)
    name = models.TextField()
    x = models.BooleanField(default=False)

    def __str__(self):
        return self.name
