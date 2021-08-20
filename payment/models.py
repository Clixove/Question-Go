from django.db import models
from django.contrib.auth.models import Group, User
from django.utils.timezone import now


class Plan(models.Model):
    name = models.CharField(max_length=64)
    price = models.FloatField()
    duration = models.IntegerField(verbose_name='subscription days')
    promotion = models.CharField(max_length=64, verbose_name='promotion slogan', blank=True)
    features = models.TextField(help_text='one line each feature', blank=True)
    groups = models.ManyToManyField(Group, blank=True, verbose_name='permitted groups')
    on_sale = models.BooleanField()

    def __str__(self):
        return self.name

    def display_features(self):
        return self.features.split('\n')


class Subscription(models.Model):
    user = models.ForeignKey(User, models.RESTRICT)
    plan = models.ForeignKey(Plan, models.RESTRICT)
    expired_time = models.DateTimeField()

    def expired(self):
        return self.expired_time < now()


class Transaction(models.Model):
    created_time = models.DateTimeField(auto_now_add=True)
    created_user = models.ForeignKey(User, models.DO_NOTHING, related_name='payment_created_user')
    plan = models.ForeignKey(Plan, models.DO_NOTHING, null=True, blank=True)

    amount = models.FloatField()
    paid = models.BooleanField()

    method = models.CharField(max_length=64, blank=True)
    token = models.CharField(max_length=16, blank=True)


def query_permitted_groups(user: User) -> None:
    for plan in Plan.objects.all():
        [user.groups.remove(g.id) for g in plan.groups.all()]
    for subscription in user.subscription_set.filter(expired_time__gt=now()):
        [user.groups.add(g.id) for g in subscription.plan.groups.all()]
