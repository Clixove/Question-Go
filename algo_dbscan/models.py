from django.db import models
from task_manager.models import Step
from library.models import Paper


class MyDBSCAN(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="dbscan_dataframe")
    knn_figure = models.TextField(blank=True)
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="dbscan_model")
    class_dict = models.TextField(blank=True)


class Column(models.Model):
    algorithm = models.ForeignKey(MyDBSCAN, models.CASCADE)
    name = models.TextField()
    x_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
