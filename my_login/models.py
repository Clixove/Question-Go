from django.db import models
from django.contrib.auth.models import Group, User


class RegisterGroup(models.Model):
    group = models.ForeignKey(Group, models.CASCADE)

    def __str__(self):
        return self.group.name


class Register(models.Model):
    username = models.CharField(max_length=150)
    password = models.CharField(max_length=150)
    email = models.EmailField()
    invitation_code = models.TextField(unique=True)
    group = models.ForeignKey(Group, models.CASCADE)

    def __str__(self):
        return self.username
