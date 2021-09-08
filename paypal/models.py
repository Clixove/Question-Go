import json

from django.contrib.auth.models import Group, User
from django.core.validators import MinValueValidator
from django.db import models
from django.utils.timezone import now

from paypalcheckoutsdk.core import PayPalHttpClient, SandboxEnvironment, LiveEnvironment
from paypalcheckoutsdk.orders import OrdersGetRequest

with open('token/paypal.json', 'r') as f:
    paypal_config = json.load(f)


class Plan(models.Model):
    name = models.CharField(max_length=64)
    badge = models.CharField(max_length=16, blank=True)  # example: 10% OFF

    permitted_groups = models.ManyToManyField(Group, blank=True)
    duration = models.IntegerField(help_text='Unit: days')
    price = models.FloatField(validators=[MinValueValidator(0)])
    currency = models.CharField(max_length=4)
    description = models.TextField(blank=True)
    on_sale = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Subscription(models.Model):
    user = models.ForeignKey(User, models.RESTRICT)
    plan = models.ForeignKey(Plan, models.RESTRICT)
    expired_time = models.DateTimeField()

    def expired(self):
        return self.expired_time < now()


class Transaction(models.Model):
    created_time = models.DateTimeField(auto_now_add=True)
    created_user = models.ForeignKey(User, models.DO_NOTHING, related_name='user_paypal')
    plan = models.ForeignKey(Plan, models.DO_NOTHING, null=True, blank=True)
    price = models.FloatField(validators=[MinValueValidator(0)])
    currency = models.CharField(max_length=4)
    token = models.CharField(max_length=32)
    paid = models.BooleanField()


class LockedGroup(models.Model):
    group = models.OneToOneField(Group, models.CASCADE)

    def __str__(self):
        return self.group.name


class PayPalClient:
    def __init__(self):
        self.client_id = paypal_config['client_id']
        self.client_secret = paypal_config['secret']

        """Set up and return PayPal Python SDK environment with PayPal access credentials.
           This sample uses SandboxEnvironment. In production, use LiveEnvironment."""

        self.environment = SandboxEnvironment(client_id=self.client_id, client_secret=self.client_secret)

        """ Returns PayPal HTTP client instance with environment that has access
            credentials context. Use this instance to invoke PayPal APIs, provided the
            credentials have access. """
        self.client = PayPalHttpClient(self.environment)

    def object_to_json(self, json_data):
        """
        Function to print all json data in an organized readable manner
        """
        result = {}
        itr = json_data.__dict__.items()
        for key, value in itr:
            # Skip internal attributes.
            if key.startswith("__"):
                continue
            if isinstance(value, list):
                result[key] = self.array_to_json_array(value)
            else:
                if not self.is_primittive(value):
                    result[key] = self.object_to_json(value)
                else:
                    result[key] = value
        return result

    def array_to_json_array(self, json_array):
        result = []
        if isinstance(json_array, list):
            for item in json_array:
                if not self.is_primittive(item):
                    result.append(self.object_to_json(item))
                else:
                    if isinstance(item, list):
                        result.append(self.array_to_json_array(item))
                    else:
                        result.append(item)
        return result

    @staticmethod
    def is_primittive(data):
        return isinstance(data, str) or isinstance(data, int)


class GetOrder(PayPalClient):
    # 2. Set up your server to receive a call from the client
    """You can use this function to retrieve an order by passing order ID as an argument"""

    def get_order(self, order_id):
        """Method to get order"""
        request = OrdersGetRequest(order_id)
        # 3. Call PayPal to get the transaction
        response = self.client.execute(request)
        # 4. Save the transaction in your database. Implement logic to save transaction to your database for future
        # reference.
        # if debug:
        #     print('Status Code: ', response.status_code)
        #     print('Status: ', response.result.status)
        #     print('Order ID: ', response.result.id)
        #     print('Intent: ', response.result.intent)
        #     print('Links:')
        #     for link in response.result.links:
        #         print(f'\t{link.rel}: {link.href}\tCall Type: {link.method}')
        #     print(f'Gross Amount: {response.result.purchase_units[0].amount.currency_code} '
        #           f'{response.result.purchase_units[0].amount.value}')
        return {
            'status_code': response.status_code,
            'status': response.result.status,
            'currency': response.result.purchase_units[0].amount.currency_code,
            'price': response.result.purchase_units[0].amount.value
        }


def query_permitted_groups(user: User) -> None:
    [user.groups.remove(g.group.id) for g in LockedGroup.objects.all()]
    for subscription in user.subscription_set.filter(expired_time__gt=now()):
        [user.groups.add(g.id) for g in subscription.plan.permitted_groups.all()]
