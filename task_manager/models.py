from django.contrib.auth.models import User
from django.db import models
from library.models import Paper


class Task(models.Model):
    user = models.ForeignKey(User, models.CASCADE)
    created_time = models.DateTimeField(auto_now_add=True)
    modified_time = models.DateTimeField(auto_now=True)
    name = models.CharField(max_length=64)

    def __str__(self):
        return self.name

    def busy(self):
        return any([x.status == 2 for x in self.step_set.all()])


class OpenedTask(models.Model):
    user = models.OneToOneField(User, models.CASCADE)
    task = models.ForeignKey(Task, models.CASCADE)

    def __str__(self):
        return self.user.username


color_picker = {1: "text-primary", 2: "text-warning", 3: "text-success", 4: "text-danger"}


class Step(models.Model):
    task = models.ForeignKey(Task, models.CASCADE)
    name = models.CharField(max_length=64)
    model_id = models.PositiveBigIntegerField()
    view_link = models.TextField()
    status = models.IntegerField(choices=[(1, "PREPARED"), (2, "RUNNING"), (3, "DONE"), (4, "INTERRUPTED")], default=1)

    linked_data = models.ForeignKey(Paper, models.CASCADE, blank=True, null=True, related_name='linked_data')
    predicted_data = models.ForeignKey(Paper, models.SET_NULL, blank=True, null=True, related_name='predicted_data')
    note = models.TextField(blank=True)
    error_message = models.TextField(blank=True)

    def __str__(self): return self.view_link

    def status_color(self): return color_picker[self.status]

    def open_permission(self, user): return any([x.user == user for x in self.task.openedtask_set.all()])
