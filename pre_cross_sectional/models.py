from django.db import models
from task_manager.models import Step
from library.models import Paper


class PreProcessing(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True,
                                  related_name="preprocessing_cross_sectional_dataframe")

    report = models.TextField(blank=True)

    note = models.TextField(blank=True)
    error_message = models.TextField(blank=True)

    def open_permission(self, user):
        return any([x.user == user for x in self.step.task.openedtask_set.all()])


class Column(models.Model):
    algorithm = models.ForeignKey(PreProcessing, models.CASCADE)
    name = models.TextField()

    def __str__(self):
        return self.name
