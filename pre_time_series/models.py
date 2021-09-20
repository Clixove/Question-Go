from django.db import models

from library.models import Paper
from task_manager.models import Step


class TimeSeries(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="ts_dataframe")
    normalizers = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="ts_normalizers")
    from_datetime = models.DateTimeField(blank=True, null=True)
    to_datetime = models.DateTimeField(blank=True, null=True)
    periods = models.IntegerField(blank=True, null=True)

    def open_permission(self, user):
        return any([x.user == user for x in self.step.task.openedtask_set.all()])

    class Meta:
        verbose_name_plural = 'Time Series'


class Sheet(models.Model):
    algorithm = models.ForeignKey(TimeSeries, models.CASCADE)
    name = models.TextField()
    is_time_series = models.BooleanField(default=False)
    is_label = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class Column(models.Model):
    algorithm = models.ForeignKey(TimeSeries, models.CASCADE)
    name = models.TextField()
    belong_time_series = models.BooleanField(default=True)
    is_date = models.BooleanField(default=False)
    is_index = models.BooleanField(default=False)
    is_label = models.BooleanField(default=False)
    use = models.BooleanField(default=False)
    log = models.BooleanField(default=False)
    diff = models.BooleanField(default=False)
    fill_na_avg = models.BooleanField(default=False)  # F: cumulative, T: avg

    def __str__(self):
        return self.name
