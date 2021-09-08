import django.db.utils
from django import forms
from django.conf import settings
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from ratelimit.decorators import ratelimit
from django.db.utils import IntegrityError
import json
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.utils import formataddr

from .models import *

from paypal.models import query_permitted_groups


for password_validator in settings.AUTH_PASSWORD_VALIDATORS:
    if password_validator['NAME'] == 'django.contrib.auth.password_validation.MinimumLengthValidator' and \
            'OPTIONS' in password_validator.keys() and \
            'min_length' in password_validator['OPTIONS'].keys():
        min_password_value = password_validator['OPTIONS']['min_length']
        break
else:
    min_password_value = 8


class LoginSheet(forms.Form):
    username = forms.CharField(max_length=64, required=True,
                               widget=forms.TextInput({"class": "form-control"}))
    password = forms.CharField(widget=forms.PasswordInput({"class": "form-control"}),
                               max_length=64, required=True)


def view_login(req):
    return render(req, "my_login/login_form.html", context={"LoginSheet": LoginSheet()})


@require_POST
@csrf_exempt
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
    query_permitted_groups(user)
    return redirect("/main")


@login_required(login_url='/main')
def delete_login(req):
    logout(req)
    return redirect('/main')


class RegisterSheet(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput({"class": "form-control"}),
        label="Username", max_length=150,
        help_text="English characters and digits (1-150) only.",
    )
    password = forms.CharField(
        widget=forms.PasswordInput({"class": "form-control"}),
        label="Password",
        min_length=min_password_value, max_length=150,
        help_text=f"English characters and digits ({min_password_value}-150) only.",
    )
    password_again = forms.CharField(
        widget=forms.PasswordInput({"class": "form-control"}),
        label="Password Again",
        min_length=min_password_value, max_length=150,
    )
    email = forms.EmailField(
        widget=forms.EmailInput({"class": "form-control"}),
    )
    try:
        group = forms.ModelChoiceField(
            RegisterGroup.objects.all(), initial=RegisterGroup.objects.first(),
            widget=forms.Select({"class": "form-select"}), empty_label=None)
    except django.db.utils.OperationalError:
        pass


def view_register(req):
    context = {
        'RegisterSheet': RegisterSheet(),
    }
    return render(req, "my_login/register.html", context)


with open('token/smtp.json', "r") as f:
    config = json.load(f)


def send_confirm_email(invitation_code: str, receiver: str, host: str):
    msg = f"""
    <p> I'm so glad you registered for Clixove software! </p>
    <p> <strong> Here's your confirmation link: </strong> 
    <a href="{host}/my_login/register/confirm/{invitation_code}">ACTIVE YOUR ACCOUNT</a> </p>
    <p> If you doesn't register any software from our company, please ignore this email. If our emails disturb you, 
    you can report the abuse to <a href="mailto:cloudy@clixove.com">cloudy@clixove.com</a>.
    It may be caused by someone's batch operation by robots. </p>
    <p>For more information about our organization: <a href="https://blog.clixove.com/"> Clixove </a></p>
    <p>Best wishes! Science will make our life better.</p>
    <p>Cloudy</p>
    <p>Developer of Clixove software</p>
    """
    msg = MIMEText(msg, 'html', 'utf-8')
    msg['From'] = formataddr(('Clixove', config['username']))
    msg['To'] = formataddr((receiver, receiver))
    msg['Subject'] = 'Clixove Registration'
    server = smtplib.SMTP_SSL(config['host'], config['port'])
    server.login(config['username'], config['password'])
    server.sendmail(config['username'], [receiver], msg.as_string())
    server.quit()


@csrf_exempt
@require_POST
@ratelimit(key='header:x-real-ip', rate='70/10m', block=True)
@ratelimit(key='post:username', rate='2/1m', block=True)
@ratelimit(key='post:email', rate='1/1m', block=True)
def add_register(req):
    register_sheet = RegisterSheet(req.POST)
    if not register_sheet.is_valid():
        return redirect('/my_login/register?message=Submission is not valid.&color=danger')
    if not register_sheet.cleaned_data['password'] == register_sheet.cleaned_data['password_again']:
        return redirect('/my_login/register?message=The twice password don\'t match.&color=danger')
    invitation_code = ''.join(random.choices(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, k=64))
    try:
        send_confirm_email(invitation_code, register_sheet.cleaned_data['email'], host=req.META['HTTP_HOST'])
    except Exception as e:
        return redirect(f'/my_login/register?message={e}&color=warning')
    if User.objects.filter(username=register_sheet.cleaned_data['username']).exists() or \
            Register.objects.filter(username=register_sheet.cleaned_data['username']).exists():
        return redirect('/my_login/register?message=This user has been registered.&color=danger')
    new_register = Register(
        username=register_sheet.cleaned_data['username'],
        password=register_sheet.cleaned_data['password'],
        email=register_sheet.cleaned_data['email'],
        invitation_code=invitation_code,
        group=register_sheet.cleaned_data['group'].group,
    )
    try:
        new_register.save()
    except IntegrityError:
        return redirect('/my_login/register/add')
    return redirect('/my_login/register?message=The email has been successfully sent.&color=success')


def add_user(req, invitation_code):
    try:
        application = Register.objects.get(invitation_code=invitation_code)
    except Register.DoesNotExist:
        return redirect('/main?message=Invitation code incorrect.&color=warning')
    new_user = User(username=application.username, email=application.email)
    new_user.set_password(application.password)
    new_user.save()
    new_user.groups.add(application.group)
    application.delete()
    return redirect('/main?message=Register successfully.&color=success')
