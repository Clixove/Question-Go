from django.contrib import admin
from .models import *


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['user', 'created_time', 'modified_time', 'name']
    list_filter = ['created_time', 'modified_time']
    search_fields = ['name']
    autocomplete_fields = ['user']


@admin.register(OpenedTask)
class DefaultTaskAdmin(admin.ModelAdmin):
    list_display = ['user', 'task']
    autocomplete_fields = ['user', 'task']


@admin.register(Step)
class StepAdmin(admin.ModelAdmin):
    list_display = ['task', 'view_link', 'status']
    list_filter = ['status']
    autocomplete_fields = ['task']
