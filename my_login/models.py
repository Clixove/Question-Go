from django.contrib.auth.models import Group
from django.db import models


class RegistryEntries(models.Model):
    registry_name = models.CharField(max_length=64, unique=True)
    groups = models.ManyToManyField(Group)

    def __str__(self):
        return self.registry_name


class InvitationCode(models.Model):
    email = models.EmailField()
    invitation_code = models.TextField(unique=True)

    def __str__(self):
        return self.email
