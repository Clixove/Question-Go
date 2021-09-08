from django.contrib import admin
from django.contrib.auth import authenticate
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType

from .models import *


@admin.register(Register)
class RegisterAdmin(admin.ModelAdmin):
    list_display = ['username', 'group', 'email']
    list_filter = ['group']
    actions = ['admit']
    exclude = ['password']
    search_fields = ['username']

    def admit(self, _, queryset):
        for application in queryset:
            new_user, created = User.objects.get_or_create(username=application.username)
            if created:
                new_user.set_password(application.password)
                new_user.save()
            else:
                new_user = authenticate(username=application.username, password=application.password)
                if not new_user:
                    application.delete()
                    continue
            new_user.email = application.email
            new_user.groups.add(application.group)
            application.delete()

    admit.short_description = "Admit"


@admin.register(RegisterGroup)
class RegisterGroupAdmin(admin.ModelAdmin):
    list_display = ['group']
    autocomplete_fields = ['group']


# Register your models here.
admin.site.site_url = "/main"
admin.site.site_header = admin.site.site_title = "Question Go"
admin.site.index_title = "Home"


@admin.register(ContentType)
class ContentTypeAdmin(admin.ModelAdmin):
    list_filter = ['app_label']


@admin.register(Permission)
class PermissionAdmin(admin.ModelAdmin):
    list_display = ['name', 'content_type', 'codename']
    list_filter = ['content_type']
