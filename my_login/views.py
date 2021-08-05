import django.db.utils
from django import forms
from django.conf import settings
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import *
from payment.models import Subscription, LockedGroup
from django.utils.timezone import now

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
    context = {"LoginSheet": LoginSheet()}
    return render(req, "my_login/login.html", context)


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
    [user.groups.remove(p.group) for p in LockedGroup.objects.all()]
    for subscription in Subscription.objects.filter(user=user, expired_time__gt=now()):
        [user.groups.add(g) for g in subscription.plan.permitted_groups.all()]
    login(req, user)
    return redirect("/task/retrieve")


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
    bio = forms.CharField(
        widget=forms.Textarea({"class": "form-control"}),
        label="Biography", required=False,
        help_text="Information that is showed to admission staffs of the group. "
                  "Max lengthen 500 characters."
    )
    try:
        group = forms.ModelChoiceField(
            Group.objects, initial=Group.objects.first(),
            widget=forms.Select({"class": "form-select"}),
        )
    except django.db.utils.OperationalError:
        pass


def view_register(req):
    context = {
        'RegisterSheet': RegisterSheet(),
    }
    return render(req, "my_login/register.html", context)


@csrf_exempt
@require_POST
def add_register(req):
    register_sheet = RegisterSheet(req.POST)
    if not register_sheet.is_valid():
        return redirect("/my_login/register?message=Submission is not valid.")
    if not register_sheet.cleaned_data['password'] == register_sheet.cleaned_data['password_again']:
        return redirect("/my_login/register?message=The twice password don't match.")
    new_register = Register(
        username=register_sheet.cleaned_data['username'],
        password=register_sheet.cleaned_data['password'],
        email=register_sheet.cleaned_data['email'],
        bio=register_sheet.cleaned_data['bio'],
        group=register_sheet.cleaned_data['group']
    )
    new_register.save()
    return redirect("/my_login/register?message=Success.&success=1")


def view_article(req, article_name):
    return render(req, f"articles/{article_name}")
