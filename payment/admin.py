from django.contrib import admin
from .models import *


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'price', 'duration']
    filter_horizontal = ['groups']
    search_fields = ['name']


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['user', 'plan', 'expired_time']
    search_fields = ['user']
    autocomplete_fields = ['user', 'plan']


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['created_time', 'created_user', 'amount', 'paid', 'plan', 'method', 'token']
    list_filter = ['created_time', 'paid', 'plan', 'method']
    autocomplete_fields = ['created_user', 'plan']
    search_fields = ['token']


@admin.register(PayingMethod)
class MethodAdmin(admin.ModelAdmin):
    list_display = ['name', 'payment_button_url']


@admin.register(Prestige)
class PrestigeAdmin(admin.ModelAdmin):
    list_display = ['created_time', 'created_user', 'amount', 'plan']
    list_filter = ['plan', 'created_time']
    autocomplete = ['created_user']


@admin.register(WebsiteManager)
class WebsiteManagerAdmin(admin.ModelAdmin):
    list_display = ['user', 'email']
    autocomplete_fields = ['user']
