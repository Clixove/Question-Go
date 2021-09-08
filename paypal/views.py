from django import forms
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http.response import HttpResponse
from .models import *
from django.contrib.auth.decorators import permission_required
from datetime import timedelta
from django.conf import settings


def view_plans(req):
    context = {
        'client_id': paypal_config['client_id'],
        'plans': Plan.objects.filter(on_sale=True),
        'payment_permission': req.user.has_perm('paypal.add_transaction'),
    }
    return render(req, "paypal/plans.html", context)


class Order(forms.Form):
    plan = forms.ModelChoiceField(Plan.objects.filter(on_sale=True), widget=forms.HiddenInput())
    order_id = forms.CharField(max_length=32, widget=forms.HiddenInput())


@csrf_exempt
@require_POST
@permission_required(
    'paypal.add_transaction',
    login_url='/paypal/plans?message=Do not have permission to add transactions.&color=danger'
)
def add_transaction(req):
    order = Order(req.POST)
    if not order.is_valid():
        return HttpResponse('/paypal/plans?message=Payment is approved by Paypal but cannot be recognized by our '
                            'server. Please contact sales for further information.&color=danger')
    checkout = GetOrder().get_order(order.cleaned_data['order_id'])
    if checkout['status_code'] != 200:
        return HttpResponse('/paypal/plans?message=Our server cannot request transaction from Paypal.&color=danger')
    if checkout['status'] != 'COMPLETED':
        new_transaction = Transaction(created_user=req.user, plan=order.cleaned_data['plan'], price=checkout['price'],
                                      currency=checkout['currency'], token=order.cleaned_data['order_id'], paid=False)
        new_transaction.save()
        return HttpResponse(f'/paypal/plans?message=The status of this transaction is {checkout["status"]}, so'
                            'it is not paid.&color=danger')
    # TODO: cannot ensure this order is paid to me.
    new_transaction = Transaction(created_user=req.user, plan=order.cleaned_data['plan'], price=checkout['price'],
                                  currency=checkout['currency'], token=order.cleaned_data['order_id'], paid=True)
    new_transaction.save()
    try:
        subscription = Subscription.objects.get(user=req.user, plan=order.cleaned_data['plan'])
    except Subscription.DoesNotExist:
        new_subscription = Subscription(user=req.user, plan=order.cleaned_data['plan'],
                                        expired_time=now() + timedelta(days=order.cleaned_data['plan'].duration))
        new_subscription.save()
        return HttpResponse('/paypal/plans?message=Payment succeed, and new subscription is added.&color=success')
    subscription.expired_time = max(now(), subscription.expired_time) + order.cleaned_data['plan'].duration
    subscription.save()
    query_permitted_groups(req.user)
    return HttpResponse('/paypal/plans?message=Payment succeed, and the subscription is extended.&color=success')


@permission_required(
    'paypal.view_transaction',
    login_url='/paypal/plans?message=Do not have permission to view transactions.&color=danger'
)
def view_transaction(req):
    context = {
        'timezone': settings.TIME_ZONE,
        'transactions': Transaction.objects.filter(created_user=req.user).order_by('-created_time'),
    }
    return render(req, 'paypal/transaction.html', context)


@permission_required(
    'paypal.view_subscription',
    login_url='/paypal/plans?message=Do not have permission to view subscriptions.&color=danger'
)
def view_subscription(req):
    context = {
        'timezone': settings.TIME_ZONE,
        'subscriptions': Subscription.objects.filter(user=req.user).order_by('-expired_time'),
    }
    return render(req, 'paypal/subscription.html', context)
