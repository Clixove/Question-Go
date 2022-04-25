import json
import random
import smtplib
import string
from email.mime.text import MIMEText
from email.utils import formataddr

from django import forms
from django.conf import settings
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from ratelimit.decorators import ratelimit

from .models import *

# Password initialization
for password_validator in settings.AUTH_PASSWORD_VALIDATORS:
    if password_validator['NAME'] == 'django.contrib.auth.password_validation.MinimumLengthValidator' and \
            'OPTIONS' in password_validator.keys() and \
            'min_length' in password_validator['OPTIONS'].keys():
        min_password_value = password_validator['OPTIONS']['min_length']
        break
else:
    min_password_value = 8
# Email initialization
with open('token/smtp.json', "r") as f:
    config = json.load(f)
# Random invitation code
max_attempt = 5


class LoginSheet(forms.Form):
    username = forms.CharField(max_length=64, required=True,
                               widget=forms.TextInput({"class": "form-control"}))
    password = forms.CharField(widget=forms.PasswordInput({"class": "form-control"}),
                               max_length=64, required=True)


class ChangePasswordSheet(forms.Form):
    old_password = forms.CharField(widget=forms.PasswordInput({"class": "form-control"}),
                                   max_length=64, required=True)
    new_password = forms.CharField(widget=forms.PasswordInput({"class": "form-control"}),
                                   max_length=64, required=True)
    new_password_again = forms.CharField(widget=forms.PasswordInput({"class": "form-control"}),
                                         max_length=64, required=True)


def view_login(req):
    return JsonResponse({
        'login_sheet': LoginSheet().as_p(),
        'change_password_sheet': ChangePasswordSheet().as_p(),
    })


@require_POST
def add_login(req):
    sheet1 = LoginSheet(req.POST)
    if not sheet1.is_valid():
        return redirect('/main?message=Login form is not valid.&color=danger')
    user = authenticate(req,
                        username=sheet1.cleaned_data['username'],
                        password=sheet1.cleaned_data['password'])
    if not user:
        return redirect('/main?message=Username or password is not correct.&color=danger')
    login(req, user)
    return redirect("/main")


@login_required(login_url='/main')
def delete_login(req):
    logout(req)
    return redirect('/main')


class InvitationCodeSheet(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput({"class": "form-control"}),
    )


def view_register(req):
    context = {
        'RegisterSheet': InvitationCodeSheet(),
    }
    return render(req, "my_login/register.html", context)


class RegistrySheet(forms.Form):
    invitation_code = forms.CharField(
        widget=forms.Textarea({'class': 'form-control', 'height': 5}),
        help_text='Invitation code is included in the email we just sent to you.'
    )
    username = forms.CharField(
        widget=forms.TextInput({"class": "form-control"}), max_length=150,
        help_text="English characters and digits (1-150) only.",
    )
    password = forms.CharField(
        widget=forms.PasswordInput({"class": "form-control"}), min_length=min_password_value, max_length=150,
        help_text=f"English characters and digits ({min_password_value}-150) only.",
    )
    password_again = forms.CharField(
        widget=forms.PasswordInput({"class": "form-control"}), min_length=min_password_value, max_length=150,
    )
    entry = forms.ModelChoiceField(
        RegistryEntries.objects.all(),
        widget=forms.Select({"class": "form-select"}), empty_label=None,
    )


@require_POST
@ratelimit(key='header:x-real-ip', rate='70/10m', block=True)
@ratelimit(key='post:email', rate='1/1m', block=True)
def add_register(req):
    register_sheet = InvitationCodeSheet(req.POST)
    if not register_sheet.is_valid():
        return redirect('/my_login/register?message=Submission is not valid.&color=danger')

    invitation_code = ''.join(random.choices(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, k=64))
    i = 1
    while InvitationCode.objects.filter(invitation_code=invitation_code).exists():
        if i > max_attempt:
            return redirect('/my_login/register?message=An internal error occurs, please try again.&color=danger')
        invitation_code = ''.join(random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits, k=64))
        i += 1
    receiver = register_sheet.cleaned_data['email']
    try:
        msg = render(req, 'my_login/email.html', {'invitation_code': invitation_code}).content.decode('utf-8')
        msg = MIMEText(msg, 'html', 'utf-8')
        msg['From'] = formataddr(('Clixove', config['username']))
        msg['To'] = formataddr((receiver, receiver))
        msg['Subject'] = 'Clixove Registration'
        server = smtplib.SMTP_SSL(config['host'], config['port'])
        server.login(config['username'], config['password'])
        server.sendmail(config['username'], [receiver], msg.as_string())
        server.quit()
    except Exception as e:
        return redirect(f'/my_login/register?message={e}&color=warning')
    new_register = InvitationCode(
        email=register_sheet.cleaned_data['email'],
        invitation_code=invitation_code,
    )
    new_register.save()
    return redirect(f'/my_login/confirm?message=The confirming email has been successfully sent.&color=success')


def view_confirm(req):
    context = {
        'invitation_code_sheet': RegistrySheet()
    }
    return render(req, 'my_login/confirm.html', context)


@require_POST
def add_user(req):
    registry = RegistrySheet(req.POST)
    if not registry.is_valid():
        return redirect('/my_login/confirm?message=Submission is not valid.&color=danger')
    try:
        application = InvitationCode.objects.get(invitation_code=registry.cleaned_data['invitation_code'])
    except InvitationCode.DoesNotExist:
        return redirect('/my_login/confirm?message=Invitation code is incorrect.&color=warning')
    try:
        entry = RegistryEntries.objects.get(registry_name=registry.cleaned_data['entry'])
    except RegistryEntries.DoesNotExist:
        return redirect('/my_login/confirm?message=Submission is not valid.&color=danger')
    if registry.cleaned_data['password'] != registry.cleaned_data['password_again']:
        return redirect('/my_login/confirm?message=The passwords inputted twice are inconsistent.&color=danger')
    if User.objects.filter(username=registry.cleaned_data['username']).exists():
        return redirect('/my_login/confirm?message=Username is occupied.&color=danger')
    new_user = User(username=registry.cleaned_data['username'], email=application.email)
    new_user.set_password(registry.cleaned_data['password'])
    new_user.save()
    [new_user.groups.add(g) for g in entry.groups.all()]
    application.delete()
    login(req, new_user)
    return redirect('/main?message=Register successfully.&color=success')


@require_POST
@login_required(login_url='/main?message=Please log in first.&color=danger')
def change_password(req):
    cpf = ChangePasswordSheet(req.POST)
    if not cpf.is_valid():
        return redirect('/main?message=Submission is not valid.&color=danger')
    user = authenticate(req,
                        username=req.user.username,
                        password=cpf.cleaned_data['old_password'])
    if not user:
        return redirect('/main?message=Password is not correct.&color=danger')
    if cpf.cleaned_data['new_password'] != cpf.cleaned_data['new_password_again']:
        return redirect('/main?message=The twice new passwords are inconsistent.&color=danger')
    user.set_password(cpf.cleaned_data['new_password'])
    user.save()
    login(req, user)
    return redirect('/main?message=Password changed successfully.&color=success')
