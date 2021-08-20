import json
import random
import smtplib
import string
from datetime import timedelta
from email.mime.text import MIMEText
from email.utils import formataddr

from django import forms
from django.contrib.auth.decorators import permission_required
from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from ratelimit.decorators import ratelimit

from .models import *

with open('token/smtp.json', "r") as f:
    config = json.load(f)


def send_notification(donation: Transaction, host: str):
    receivers = [x.email for x in WebsiteManager.objects.all()]
    if len(receivers) == 0:
        return
    msg = f"""
    <p style="font-size: large"> New donation </p>
    <p> Host: {host} </p>
    <p> Donation code: {donation.token} </p>
    <p> Payment method: {donation.method} </p>
    <p> Created at: {donation.created_time} </p>
    <p> <strong> If you have received the donation, click: </strong> 
    <a href="{host}/payment/donation/prestige/add/{donation.token}">CONFIRM</a> </p>
    <p> Please check the payment and add prestige points for the user soon. </p>
    <p> You receive this email because you're listed as a manager of website: {host} </p>
    <hr>
    <p>For more information about our organization: <a href="https://blog.clixove.com/"> Clixove </a></p>
    <p>Best wishes! Science will make our life better.</p>
    <p>Cloudy</p>
    <p>Developer of Clixove software</p>
        """
    msg = MIMEText(msg, 'html', 'utf-8')
    msg['From'] = formataddr(('Clixove', config['username']))
    msg['To'] = ', '.join(receivers)
    msg['Subject'] = 'New donation from Clixove website'
    server = smtplib.SMTP_SSL(config['host'], config['port'])
    server.login(config['username'], config['password'])
    server.sendmail(config['username'], receivers, msg.as_string())
    server.quit()


class DonationMethod(forms.Form):
    method = forms.ModelChoiceField(
        PayingMethod.objects.all(), widget=forms.Select({'class': 'form-select'}),
        empty_label=None, label='')
    token = forms.CharField(max_length=16, widget=forms.HiddenInput())


class DonationToken(forms.Form):
    token = forms.CharField(max_length=16, widget=forms.HiddenInput())


class DonationConfirm(forms.Form):
    amount = forms.FloatField(
        widget=forms.NumberInput({'class': 'form-control'}), help_text='Converted to USD amount.'
    )
    token = forms.CharField(max_length=16, widget=forms.HiddenInput())
    conclusion = forms.ChoiceField(
        choices=[(True, 'Yes, this token has paid.'), (False, 'No, this token is not paid.')],
        required=False, initial=True, widget=forms.Select({"class": "form-select"}),
    )


def view_plan(req):
    context = {
        'plans': Plan.objects.filter(on_sale=True),
    }
    return render(req, 'payment/donation/plans.html', context)


def donate(req):
    donation_code = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=16))
    context = {
        'donation_method': DonationMethod(initial={'method': PayingMethod.objects.first(), 'token': donation_code}),
        'donation_token': DonationToken(initial={'token': donation_code}),
        'token': donation_code,
    }
    return render(req, 'payment/donation/donate.html', context)


@permission_required(
    'payment.view_transaction',
    login_url='/payment/donation/plans?message=Do not have permission to view transactions.&color=danger')
def view_transaction(req):
    context = {
        'transactions': Transaction.objects.filter(created_user=req.user).order_by('-created_time'),
    }
    return render(req, 'payment/transaction.html', context)


@csrf_exempt
@require_POST
@permission_required(
    'payment_donation.view_payingmethod',
    login_url='/payment/donation/plans?message=Do not have permission to view paying methods.&color=danger')
def add_transaction_1(req):
    dm = DonationMethod(req.POST)
    if not dm.is_valid():
        return HttpResponse(f'Donation form is not valid: <br> {dm.errors}')
    method = dm.cleaned_data['method']
    new_donation = Transaction(created_user=req.user, amount=0, paid=False, method=method.name,
                               token=dm.cleaned_data['token'])
    new_donation.save()
    return redirect(method.payment_button_url)


@csrf_exempt
@require_POST
@permission_required(
    'payment.add_transaction',
    login_url='/payment/donation?message=Do not have permission to add transactions.&color=danger')
@ratelimit(key='header:x-real-ip', rate='70/10m', block=True)
@ratelimit(key='post:token', rate='1/1m', block=True)
def add_transaction_2(req):
    dt = DonationToken(req.POST)
    if not dt.is_valid():
        return redirect('/payment/donation?message=Notification fails to be sent.&color=danger')
    try:
        donation = Transaction.objects.get(created_user=req.user, paid=False, token=dt.cleaned_data['token'])
    except Transaction.DoesNotExist:
        return redirect('/payment/donation?message=Donation not found. Please click \'start\' button first.'
                        '&color=danger')
    donation.paid = True
    donation.save()
    send_notification(donation, host=req.META['HTTP_HOST'])
    return redirect('/payment/donation/transaction?message=Thank you for your donation! The '
                    'website managers will check it in several days.&color=success')


@permission_required(
    'payment_donation.view_prestige',
    login_url='/payment/donation/plans?message=Do not have permission to view prestige.&color=danger')
def view_prestige(req):
    context = {
        'prestige_s': Prestige.objects.filter(created_user=req.user),
        'my_coins': deposit(req.user),
    }
    return render(req, 'payment/donation/prestige.html', context)


@permission_required(
    'payment_donation.add_prestige',
    login_url='/payment/donation?message=Do not have permission to add prestige.&color=danger')
def view_add_prestige(req, token):
    context = {
        'donation_confirm_sheet': DonationConfirm(initial={'amount': 0, 'token': token})
    }
    return render(req, 'payment/donation/confirm.html', context)


@csrf_exempt
@require_POST
@permission_required(
    'payment_donation.add_prestige',
    login_url='/payment/donation?message=Do not have permission to add prestige.&color=danger')
def add_prestige(req):
    dc = DonationConfirm(req.POST)
    if not dc.is_valid():
        return redirect('/main?message=Confirming request is not valid.&color=danger')
    try:
        donation = Transaction.objects.get(paid=True, token=dc.cleaned_data['token'])
    except Transaction.DoesNotExist:
        return redirect('/main?message=Donation not found.&color=danger')
    if dc.cleaned_data['conclusion'] is True:
        donation.amount = dc.cleaned_data['amount']
        donation.token = ''
        donation.save()
        new_prestige = Prestige(created_user=donation.created_user, amount=dc.cleaned_data['amount'],
                                transaction=donation)
        new_prestige.save()
        return redirect('/main?message=Successfully confirm the donation.&color=success')
    else:
        donation.paid = False
        donation.save()
        return redirect('/main?message=Successfully deny the donation.&color=success')


@permission_required(
    'payment.view_subscription',
    login_url='/payment/donation/plans?message=Do not have permission to view subscription.&color=danger')
def view_subscription(req):
    context = {
        'subscriptions': Subscription.objects.filter(user=req.user).order_by('-expired_time')
    }
    return render(req, 'payment/subscription.html', context)


@permission_required(
    'payment.add_subscription',
    login_url='/payment/donation/plans?message=Do not have permission to add subscription.&color=danger')
def add_subscription(req, plan_id):
    try:
        plan = Plan.objects.get(id=plan_id)
    except Plan.DoesNotExist:
        return redirect('/payment/donation/plans?message=Plan does not exist.&color=danger')
    if deposit(req.user) < plan.price:
        return redirect('/payment/donation/plans?message=Prestige cannot afford this plan.&color=danger')
    new_prestige = Prestige(created_user=req.user, amount=-plan.price, plan=plan)
    new_prestige.save()
    try:
        subscription = Subscription.objects.get(user=req.user, plan=plan)
    except Subscription.DoesNotExist:
        new_subscription = Subscription(user=req.user, plan=plan, expired_time=now() + timedelta(days=plan.duration))
        new_subscription.save()
        return redirect('/payment/donation/plans?message=New subscription successfully added.&color=success')
    subscription.expired_time = max(now(), subscription.expired_time) + plan.duration
    subscription.save()
    return redirect('/payment/donation/plans?message=This subscription successfully extended.&color=success')
