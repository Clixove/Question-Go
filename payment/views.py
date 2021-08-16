"""
Acknowledgement:
https://github.com/fzlee/alipay is forked as payment.alipay
"""
import requests
from django import forms
from django.contrib.auth.decorators import permission_required
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.utils.timezone import timedelta
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from payment.alipay import AliPay
from .models import *

with open("token/app_private_key.pem", "r") as f:
    app_private_key = f.read()
with open("token/alipay_public_key.pem", "r") as f:
    alipay_public_key = f.read()
with open("token/app_id", "r") as f:
    app_id = f.read()

alipay = AliPay(
    appid=app_id,
    app_private_key_string=app_private_key,
    alipay_public_key_string=alipay_public_key,
    sign_type="RSA2",
)


class CurrentProduct(forms.Form):
    product_line = forms.ModelChoiceField(ProductLine.objects)


class CreateOrder(forms.Form):
    required_css_class = 'required'
    plan = forms.ModelChoiceField(Plan.objects.filter(on_sale=True), widget=forms.HiddenInput())
    redeem = forms.CharField(max_length=64, widget=forms.TextInput({"class": "form-control"}),
                             required=False, label="Redeem (Optional)")


def view_subscriptions(req):
    cp = CurrentProduct(req.GET)
    if not cp.is_valid():
        if ProductLine.objects.count() == 0:
            context = dict()
            return render(req, "payment/subscriptions.html", context)
        else:
            this_product = ProductLine.objects.first()
    else:
        this_product = cp.cleaned_data['product_line']
    context = {
        "product_lines": ProductLine.objects.all(),
        "product": this_product,
        "plans": Plan.objects.filter(product_line=this_product, on_sale=True),
        "create_order": CreateOrder(),
    }
    return render(req, "payment/subscriptions.html", context)


@permission_required("payment.view_transaction",
                     login_url="/payment?message=No permission to view transactions.&color=danger")
def view_transactions(req):
    context = {
        "transactions": Transaction.objects.filter(created_user=req.user).order_by("-created_time"),
        "subscriptions": Subscription.objects.filter(user=req.user, expired_time__gt=now()).order_by("expired_time"),
    }
    return render(req, "payment/transaction.html", context)


@permission_required("payment.view_transaction",
                     login_url="/payment?message=No permission to view transactions.&color=danger")
def view_transaction_paying_page(req, transaction_id):
    try:
        transaction = Transaction.objects.get(id=transaction_id, created_user=req.user)
    except Transaction.DoesNotExist:
        return redirect("/payment/transactions?message=This transaction does not exist.&color=danger")
    return HttpResponse(transaction.returned_page)


def add_transaction_alipay(req):
    co = CreateOrder(req.POST)
    if not co.is_valid():
        return redirect("/payment?message=Submission is not valid.&color=danger")
    plan = co.cleaned_data['plan']
    amount = plan.price
    if co.cleaned_data['redeem']:
        try:
            redeem = Redeem.objects.get(name=co.cleaned_data['redeem'], plan=plan)
        except Redeem.DoesNotExist:
            return redirect("/payment?message=The redeem does not exist.&color=danger")
        if redeem.used_up:
            return redirect("/payment?message=The redeem is used up.&color=danger")
        amount -= redeem.money_saved
    biz_content = {
        "out_trade_no": "qgo_" + now().strftime("%Y%m%d%H%M%S%f"),
        "product_code": "FAST_INSTANT_TRADE_PAY",
        "total_amount": round(max(amount, 0.01), 2),
        "subject": plan.product_line.name + " " + plan.name,
    }
    try:
        pay = alipay.client_api(
            "alipay.trade.page.pay", biz_content=biz_content,
            return_url=f"{req.scheme}://{req.META['HTTP_HOST']}/payment/transactions"
        )
        order = requests.get("https://openapi.alipay.com/gateway.do?" + pay)
    except Exception as e:
        return redirect(f"/payment?message={e}&color=danger")
    if order.status_code != 200:
        return redirect(f"/payment?message=Request to alipay server fails, Internet code {order.status_code}."
                        "&color=danger")
    if 'out_trade_no' not in biz_content.keys():
        return redirect(f"/payment?message=The alipay server doesn't return trade number.&color=danger")
    new_transaction = Transaction(
        created_user=req.user, amount=amount,
        trade_number=biz_content['out_trade_no'],
        returned_page=order.text, paid=False, plan=plan,
    )
    new_transaction.save()
    return redirect(f"/payment/transactions")


@csrf_exempt
@require_POST
@permission_required("payment.add_transaction",
                     login_url="/payment?message=No permission to add transactions.&color=danger")
def add_transaction(req):
    if req.POST['method'] == 'Alipay':
        return add_transaction_alipay(req)
    else:
        return redirect('/payment?message=Payment method does not exist.&color=warning')


@permission_required("payment.change_transaction",
                     login_url="/payment?message=No permission to change transactions.&color=danger")
def change_transaction(req, transaction_id):
    try:
        transaction = Transaction.objects.get(id=transaction_id, created_user=req.user)
    except Transaction.DoesNotExist:
        return redirect("/payment/transactions?message=This transaction does not exist.&color=danger")
    try:
        response = alipay.server_api("alipay.trade.query", biz_content={"out_trade_no": transaction.trade_number})
    except Exception as e:
        return redirect(f"/payment/transactions?message=Request fails: {e}&color=danger")
    if not (isinstance(response, dict) and "trade_status" in response.keys()):
        return redirect("/payment/transactions?message=Cannot obtain trading status.&color=danger")
    if response['trade_status'] == "TRADE_SUCCESS" and (not transaction.paid):
        transaction.paid = True
        transaction.save()
    else:
        return redirect("/payment/transactions?message=Not paid, current trading "
                        f"status is {response['trade_status']}&color=warning")
    existed_subscriptions = Subscription.objects.filter(
        user=req.user, plan__product_line=transaction.plan.product_line, expired_time__gt=now())
    for subscription in existed_subscriptions:
        subscription.expired_time += timedelta(days=transaction.plan.subscription_days)
        subscription.save()
    same_plan_subscriptions = existed_subscriptions.filter(plan=transaction.plan)
    if not same_plan_subscriptions.exists():
        new_subscription = Subscription(
            user=req.user, plan=transaction.plan,
            expired_time=now() + timedelta(days=transaction.plan.subscription_days)
        )
        new_subscription.save()
    return redirect("/payment/transactions?message=Update successfully.&color=success")
