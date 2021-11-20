from django.db import models
from task_manager.models import Step
from library.models import Paper


class MyPCA(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="pca_dataframe")
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="pca_model")
    evr_figure = models.TextField(blank=True)


class Column(models.Model):
    algorithm = models.ForeignKey(MyPCA, models.CASCADE)
    name = models.TextField()
    x_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
