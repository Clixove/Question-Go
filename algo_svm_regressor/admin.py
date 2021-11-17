from django.contrib import admin
from .models import *


@admin.register(BayesSvmRegressor)
class SVRAdmin(admin.ModelAdmin):
    pass
