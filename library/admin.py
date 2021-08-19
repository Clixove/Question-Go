from django.contrib import admin
from .models import *


@admin.register(UserStorage)
class UserStorageAdmin(admin.ModelAdmin):
    list_display = ['user', 'specific_storage']
    autocomplete_fields = ['user']


@admin.register(GroupStorage)
class PermittedStorageAdmin(admin.ModelAdmin):
    list_display = ['group', 'user_init_storage']
    autocomplete_fields = ['group']


@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'file']
    list_filter = ['user', 'role']
    autocomplete_fields = ['user']
