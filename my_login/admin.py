from django.contrib import admin
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType

from .models import *

admin.site.site_url = "/main"
admin.site.site_header = admin.site.site_title = "Analyst"
admin.site.index_title = "Home"


@admin.register(InvitationCode)
class InvitationCodeAdmin(admin.ModelAdmin):
    list_display = ['email', 'invitation_code']


@admin.register(RegistryEntries)
class RegisterEntriesAdmin(admin.ModelAdmin):
    list_display = ['registry_name']
    filter_horizontal = ['groups']


@admin.register(ContentType)
class ContentTypeAdmin(admin.ModelAdmin):
    list_filter = ['app_label']


@admin.register(Permission)
class PermissionAdmin(admin.ModelAdmin):
    list_display = ['name', 'content_type', 'codename']
    list_filter = ['content_type']
