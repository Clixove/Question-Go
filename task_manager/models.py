from django.contrib.auth.models import User
from django.db import models


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


class Step(models.Model):
    task = models.ForeignKey(Task, models.CASCADE)
    name = models.CharField(max_length=64)
    view_link = models.TextField(blank=True)
    status = models.IntegerField(choices=[(1, "NOT STARTED"), (2, "RUNNING"), (3, "DONE"), (4, "INTERRUPTED")], default=1)

    def __str__(self):
        return self.view_link

    def status_color(self):
        color_picker = {
            1: "text-primary", 2: "text-warning", 3: "text-success", 4: "text-danger"
        }
        return color_picker[self.status]
