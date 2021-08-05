"""question_go_v2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
import my_login.views as v1
import task_manager.views as v2
import payment.views as v3
import library.views as v4
import algo_linear_regression.views as v5
import pre_norm.views as v6
import pre_cross_sectional.views as v7

urlpatterns = [
    path('admin/', admin.site.urls),
    # my login
    path('main/', v1.view_login),
    path('my_login/login', v1.add_login),
    path('my_login/quit', v1.delete_login),
    path('my_login/register', v1.view_register),
    path('my_login/register/add', v1.add_register),
    path('articles/<article_name>', v1.view_article),
    # task manager
    path('task/instances', v2.view_instances),
    path('task/<int:task_id>', v2.view_task),
    path('task/open/<int:task_id>', v2.open_task),
    path('task/close', v2.close_task),
    path('task/delete/<int:task_id>', v2.delete_task),
    path('task/rename/<int:task_id>', v2.rename_task),
    path('task/new', v2.view_add_task),
    path('task/add', v2.add_task),
    path('task/retrieve', v2.retrieve_task),
    path('step/delete/<int:step_id>', v2.delete_step),
    # payment
    path('payment', v3.view_subscriptions),
    path('payment/transactions', v3.view_transactions),
    path('payment/pay/<int:transaction_id>', v3.view_transaction_paying_page),
    path('payment/add', v3.add_transaction),
    path('payment/query/<int:transaction_id>', v3.change_transaction),
    # library
    path('library', v4.view_library),
    path('library/paper/add', v4.add_paper),
    path('library/paper/delete', v4.delete_paper),
    path('library/paper/<int:paper_id>', v4.view_paper),
    path('library/paper/rename', v4.rename_paper),
    # algorithm: linear regression
    path('algo_linear_regression/add', v5.add_lr),
    path('algo_linear_regression/<int:algo_id>', v5.view_lr),
    path('algo_linear_regression/change-note', v5.change_note),
    path('algo_linear_regression/search-data', v5.search_data),
    path('algo_linear_regression/use-data', v5.use_data),
    path('algo_linear_regression/clear-data/<int:algo_id>', v5.clear_data),
    path('algo_linear_regression/confirm-error/<int:algo_id>', v5.confirm_error),
    path('algo_linear_regression/variables', v5.set_variables),
    path('algo_linear_regression/clear-variables/<int:algo_id>', v5.clear_variables),
    path('algo_linear_regression/train-model', v5.train_model),
    path('algo_linear_regression/clear-model/<int:algo_id>', v5.clear_model),
    path('algo_linear_regression/regression-line', v5.regression_line),
    path('algo_linear_regression/predict', v5.predict),
    path('algo_linear_regression/clear-predict/<int:algo_id>', v5.clear_predict),
    # pre-processing: cross-sectional data
    path('pre_cross_sectional/add', v7.add_csp),
    path('pre_cross_sectional/<int:algo_id>', v7.view_csp),
    path('pre_cross_sectional/change-note', v7.change_note),
    path('pre_cross_sectional/search-data', v7.search_data),
    path('pre_cross_sectional/use-data', v7.use_data),
    path('pre_cross_sectional/clear-data/<int:algo_id>', v7.clear_data),
    path('pre_cross_sectional/confirm-error/<int:algo_id>', v7.confirm_error),
    path('pre_cross_sectional/profile/generate', v7.generate_profile),
    path('pre_cross_sectional/profile/<int:algo_id>', v7.view_profile),
    path('pre_cross_sectional/action/<str:form_name>', v7.preprocessing_wrapper),
    # pre-processing: normalization
    path('pre_norm/add', v6.add_norm),
    path('pre_norm/<int:algo_id>', v6.view_norm),
    path('pre_norm/change-note', v6.change_note),
    path('pre_norm/search-data', v6.search_data),
    path('pre_norm/use-data', v6.use_data),
    path('pre_norm/clear-data/<int:algo_id>', v6.clear_data),
    path('pre_norm/confirm-error/<int:algo_id>', v6.confirm_error),
    path('pre_norm/train', v6.train),
    path('pre_norm/clear-model/<int:algo_id>', v6.clear_model),
    path('pre_norm/predict', v6.predict),
    path('pre_norm/clear-predict/<int:algo_id>', v6.clear_predict),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
