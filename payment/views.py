from django.shortcuts import render
from .models import *

with open('token/paypal/client_id', 'r') as f:
    client_id = f.read()


def view_plans(req):
    context = {
        'plans': Plan.objects.all(), 'client_id': client_id,
    }
    return render(req, 'payment/plans.html', context)
