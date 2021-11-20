from django.db import models
from task_manager.models import Step
from library.models import Paper


class MyKMeans(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="k_means_dataframe")
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="k_means_model")
    class_dict = models.TextField(blank=True)


class Column(models.Model):
    algorithm = models.ForeignKey(MyKMeans, models.CASCADE)
    name = models.TextField()
    x_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
