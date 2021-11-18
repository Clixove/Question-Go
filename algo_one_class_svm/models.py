from django.db import models
from library.models import Paper
from task_manager.models import Step


class BayesOneClassSVM(models.Model):
    step = models.ForeignKey(Step, models.CASCADE, blank=True, null=True)
    dataframe = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="ocs_dataframe")
    model = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name="ocs_model")
    mode = models.CharField(choices=[("5_fold", "5 fold cross validation"),
                                     ("split", "Random split to 80% training set, 20% validation set"),
                                     ("full_train", "Applying all samples for training")],
                            blank=True, max_length=10)
    hyper_parameters = models.TextField(blank=True)
    class_list = models.TextField(blank=True)
    abnormal_class_name = models.TextField(blank=True)
    confusion_matrix = models.TextField(blank=True)
    support_vectors = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True,
                                        related_name="ocs_support_vectors")


class Column(models.Model):
    algorithm = models.ForeignKey(BayesOneClassSVM, models.CASCADE)
    name = models.TextField()
    x_column = models.BooleanField(default=False)
    y_column = models.BooleanField(default=False)

    def __str__(self):
        return self.name
