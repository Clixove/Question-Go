from django.db import models
from django.contrib.auth.models import Group, User
from django.db.models import QuerySet
from django.utils.timezone import now


class Plan(models.Model):
    name = models.CharField(max_length=64)
    price = models.FloatField()
    duration = models.IntegerField(verbose_name='subscription days')
    promotion = models.CharField(max_length=64, verbose_name='promotion slogan', blank=True)
    features = models.TextField(help_text='one line each feature', blank=True)
    groups = models.ManyToManyField(Group, blank=True, verbose_name='permitted groups')

    def __str__(self):
        return self.name

    def display_features(self):
        return self.features.split('\n')


class Subscription(models.Model):
    user = models.ForeignKey(User, models.RESTRICT)
    plan = models.ForeignKey(Plan, models.RESTRICT)
    expired_time = models.DateTimeField()


class Transaction(models.Model):
    created_time = models.DateTimeField(auto_now_add=True)
    created_user = models.ForeignKey(User, models.DO_NOTHING)
    amount = models.FloatField()
    paid = models.BooleanField()
    plan = models.ForeignKey(Plan, models.DO_NOTHING)


def query_permitted_groups(user: User) -> None:
    basic_groups = QuerySet()
    for plan in Plan.objects.all():
        basic_groups = basic_groups | plan.groups.all
    basic_groups = Group.objects.exclude(basic_groups)
    granted_groups = QuerySet()
    for plan in user.subscription_set.filter(expired_time__gt=now()):
        granted_groups = granted_groups | plan.groups.all
    user.groups.remove(basic_groups)
    user.groups.add(granted_groups)
