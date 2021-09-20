from django.db import models
from task_manager.models import Step
from library.models import Paper


class BayesRfRegressor(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="rfr_dataframe")
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="rfr_model")
    training_history = models.TextField(blank=True)
    error_measure = models.TextField(blank=True)
    mode = models.CharField(choices=[("5_fold", "5 fold cross validation"),
                                     ("split", "Random split to 80% training set, 20% validation set"),
                                     ("full_train", "Applying all samples for training")],
                            blank=True, max_length=10)
    hyper_parameters = models.TextField(blank=True)
    feature_importance = models.TextField(blank=True)


class Column(models.Model):
    algorithm = models.ForeignKey(BayesRfRegressor, models.CASCADE)
    name = models.TextField()
    x_column = models.BooleanField(default=False)
    y_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
