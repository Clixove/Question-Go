from django.contrib.auth.models import User

from payment.models import *


class PayingMethod(models.Model):
    name = models.CharField(max_length=64)
    payment_button_url = models.TextField()

    def __str__(self):
        return self.name


class Prestige(models.Model):
    created_time = models.DateTimeField(auto_now_add=True)
    created_user = models.ForeignKey(User, models.DO_NOTHING, related_name='payment_donation_created_user')
    amount = models.FloatField()
    plan = models.ForeignKey(Plan, models.DO_NOTHING, blank=True, null=True)
    transaction = models.ForeignKey(Transaction, models.DO_NOTHING, blank=True, null=True)


def deposit(user: User) -> float:
    return sum([x.amount for x in Prestige.objects.filter(created_user=user)])


class WebsiteManager(models.Model):
    user = models.ForeignKey(User, models.CASCADE)
    email = models.EmailField()

    def __str__(self):
        return self.user.username
