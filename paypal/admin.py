from django.contrib import admin
from .models import *


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'duration', 'price', 'currency', 'on_sale']
    filter_horizontal = ['permitted_groups']
    list_filter = ['on_sale', 'currency']
    search_fields = ['name']


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['user', 'plan', 'expired_time']
    list_filter = ['plan', 'expired_time']
    autocomplete_fields = ['user', 'plan']
    search_fields = ['user']


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['created_time', 'created_user', 'plan', 'price', 'currency', 'paid']
    list_filter = ['created_time', 'plan', 'currency', 'paid']
    autocomplete_fields = ['created_user', 'plan']
    search_fields = ['created_user']


@admin.register(LockedGroup)
class LockedGroupAdmin(admin.ModelAdmin):
    autocomplete_fields = ['group']
