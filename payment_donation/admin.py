from django.contrib import admin
from .models import *


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
