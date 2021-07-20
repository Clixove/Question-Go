from django.db import models
from django.contrib.auth.models import User, Group
from django.utils.timezone import now


class LockedGroup(models.Model):
    group = models.OneToOneField(Group, on_delete=models.CASCADE)

    def __str__(self):
        return self.group.name


class ProductLine(models.Model):
    name = models.CharField(max_length=64)
    coin = models.CharField(max_length=4, default="CNY")

    def __str__(self):
        return self.name


class Plan(models.Model):
    name = models.CharField(max_length=64)
    subscription_days = models.PositiveIntegerField()
    price = models.FloatField()
    feature_items = models.TextField(blank=True)
    badge = models.CharField(max_length=16, blank=True)
    on_sale = models.BooleanField()
    product_line = models.ForeignKey(ProductLine, models.CASCADE)
    permitted_groups = models.ManyToManyField(Group, blank=True)

    def __str__(self):
        return self.name

    @property
    def feature_list(self):
        return self.feature_items.split("\n")


class Redeem(models.Model):
    name = models.CharField(max_length=64)
    released_amount = models.PositiveIntegerField(default=0)
    used_amount = models.PositiveIntegerField(default=0)
    plan = models.ForeignKey(Plan, models.CASCADE)
    money_saved = models.FloatField()

    @property
    def used_up(self):
        return self.used_amount >= self.released_amount

    def __str__(self):
        return self.name


class Feature(models.Model):
    name = models.TextField()
    product_line = models.ForeignKey(ProductLine, models.CASCADE)
    contained_by = models.ManyToManyField(Plan, blank=True)

    def contained_by_display(self):
        return "; ".join([x.name for x in self.contained_by.all()])

    def __str__(self):
        return self.name


class Subscription(models.Model):
    user = models.ForeignKey(User, models.RESTRICT)
    plan = models.ForeignKey(Plan, models.RESTRICT)
    expired_time = models.DateTimeField(default=now)

    def __str__(self):
        return self.user.username


class Transaction(models.Model):
    created_time = models.DateTimeField(auto_now_add=True)
    created_user = models.ForeignKey(User, models.DO_NOTHING)
    amount = models.FloatField()
    trade_number = models.CharField(max_length=64)
    returned_page = models.TextField()
    paid = models.BooleanField()
    plan = models.ForeignKey(Plan, models.DO_NOTHING)

    def __str__(self):
        return self.trade_number
