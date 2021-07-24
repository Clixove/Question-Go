from django.contrib import admin
from .models import *
from admin_site_controller.admin import RestrictedAdmin


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'product_line', 'subscription_days', 'price', 'badge', 'on_sale']
    list_filter = ['on_sale', 'product_line']
    filter_horizontal = ['permitted_groups']
    search_fields = ['name']


@admin.register(LockedGroup)
class LockedGroupAdmin(admin.ModelAdmin):
    list_display = ['group']
    autocomplete_fields = ['group']


@admin.register(Redeem)
class RedeemAdmin(admin.ModelAdmin):
    list_display = ['name', 'released_amount', 'used_amount', 'plan', 'money_saved']
    list_filter = ['plan']
    search_fields = ['name']
    autocomplete_fields = ['plan']


@admin.register(ProductLine)
class ProductLineAdmin(admin.ModelAdmin):
    list_display = ['name', 'coin']
    list_filter = ['coin']


@admin.register(Feature)
class FeatureAdmin(admin.ModelAdmin):
    list_display = ['product_line', 'name', 'contained_by_display']
    list_filter = ['product_line']
    filter_horizontal = ['contained_by']


@admin.register(Subscription)
class SubscriptionAdmin(RestrictedAdmin):
    list_display = ['user', 'plan', 'expired_time']
    list_filter = ['plan', 'expired_time']
    search_fields = ['user__username']
    autocomplete_fields = ['user', 'plan']


@admin.register(Transaction)
class TransactionAdmin(RestrictedAdmin):
    list_display = ['created_time', 'created_user', 'amount', 'trade_number', 'paid', 'plan']
    list_filter = ['created_time', 'paid', 'plan']
    search_fields = ['trade_number']
