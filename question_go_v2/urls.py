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
import paypal.views as v3
import library.views as v4
import algo_linear_regression.views as v5
import pre_norm.views as v6
import pre_cross_sectional.views as v7
import pre_time_series.views as v8
import pre_resampling.views as v19
import algo_rf_classifier.views as v9
import algo_rf_regressor.views as v10
import algo_svm_classifier.views as v11
import algo_svm_regressor.views as v12
import algo_one_class_svm.views as v13
import algo_logistic_regression.views as v14
import algo_elastic_net.views as v15
import algo_dbscan.views as v16
import algo_kmeans.views as v17
import algo_pca.views as v18

urlpatterns = [
    path('admin/', admin.site.urls),
    # my login
    path('my_login/view', v1.view_login),
    path('my_login/register', v1.view_register),
    path('my_login/add', v1.add_login),
    path('my_login/register/add', v1.add_register),
    path('my_login/confirm', v1.view_confirm),
    path('my_login/confirm/add', v1.add_user),
    path('my_login/change_password', v1.change_password),
    path('my_login/delete', v1.delete_login),
    # task manager
    path('main', v2.view_main_page),
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
    path('step/note', v2.change_note),
    path('step/confirm-error/<int:step_id>', v2.confirm_error),
    path('step/data/delete/<int:step_id>', v2.delete_data),
    path('step/data/search', v2.search_data),
    path('step/predicted/delete/<int:step_id>', v2.delete_predicted),
    # paypal
    path('paypal/refund', v3.view_refund_policy),
    path('paypal/plans', v3.view_plans),
    path('paypal/transaction/add', v3.add_transaction),
    path('paypal/transaction', v3.view_transaction),
    path('paypal/subscription', v3.view_subscription),
    # library
    path('library', v4.view_library),
    path('library/paper/add', v4.add_paper),
    path('library/paper/delete', v4.delete_paper),
    path('library/paper/<int:paper_id>', v4.view_paper),
    path('library/paper/rename', v4.rename_paper),
    # algorithm: linear regression
    path('algo_linear_regression/add', v5.add_lr),
    path('algo_linear_regression/<int:algo_id>', v5.view_lr),
    path('algo_linear_regression/import', v5.import_data),
    path('algo_linear_regression/variables', v5.set_variables),
    path('algo_linear_regression/clear-variables/<int:algo_id>', v5.clear_variables),
    path('algo_linear_regression/train-model', v5.train_model),
    path('algo_linear_regression/clear-model/<int:algo_id>', v5.clear_model),
    path('algo_linear_regression/regression-line', v5.regression_line),
    path('algo_linear_regression/predict', v5.predict),
    # pre-processing: normalization
    path('pre_norm/add', v6.add_norm),
    path('pre_norm/<int:algo_id>', v6.view_norm),
    path('pre_norm/import', v6.import_data),
    path('pre_norm/train', v6.train),
    path('pre_norm/clear-model/<int:algo_id>', v6.clear_model),
    path('pre_norm/transform', v6.transform),
    path('pre_norm/inverse-transform', v6.inverse_transform),
    # pre-processing: cross-sectional data
    path('pre_cross_sectional/add', v7.add_csp),
    path('pre_cross_sectional/<int:algo_id>', v7.view_csp),
    path('pre_cross_sectional/import', v7.import_data),
    path('pre_cross_sectional/profile/generate', v7.generate_profile),
    path('pre_cross_sectional/profile/<int:algo_id>', v7.view_profile),
    path('pre_cross_sectional/action/<str:form_name>', v7.preprocessing_wrapper),
    # pre-processing: time series
    path('pre_ts/add', v8.add_ts),
    path('pre_ts/<int:algo_id>', v8.view_ts),
    path('pre_ts/import', v8.import_data),
    path('pre_ts/select-sheet', v8.select_sheet),
    path('pre_ts/clear-sheet/<int:algo_id>', v8.clear_sheet),
    path('pre_ts/transform', v8.transform),
    path('pre_ts/clear-transform/<int:algo_id>', v8.clear_transform),
    # pre-processing: re-sampling
    path('pre_resampling/add', v19.add_resampling),
    path('pre_resampling/<int:algo_id>', v19.view_resampling),
    path('pre_resampling/import', v19.import_data),
    path('pre_resampling/variables', v19.set_variables),
    path('pre_resampling/clear-variables/<int:algo_id>', v19.clear_variables),
    path('pre_resampling/train-model', v19.train_model),
    # algorithm: random forest classifier (template for classification)
    path('algo_rf_classifier/add', v9.add_rf_classifier),
    path('algo_rf_classifier/<int:algo_id>', v9.view_rf_classifier),
    path('algo_rf_classifier/import', v9.import_data),
    path('algo_rf_classifier/variables', v9.set_variables),
    path('algo_rf_classifier/clear-variables/<int:algo_id>', v9.clear_variables),
    path('algo_rf_classifier/train-model', v9.train_model),
    path('algo_rf_classifier/clear-model/<int:algo_id>', v9.clear_model),
    path('algo_rf_classifier/predict', v9.predict),
    # algorithm: random forest regressor (template for regression)
    path('algo_rf_regressor/add', v10.add_rf_regressor),
    path('algo_rf_regressor/<int:algo_id>', v10.view_rf_regressor),
    path('algo_rf_regressor/import', v10.import_data),
    path('algo_rf_regressor/variables', v10.set_variables),
    path('algo_rf_regressor/clear-variables/<int:algo_id>', v10.clear_variables),
    path('algo_rf_regressor/train-model', v10.train_model),
    path('algo_rf_regressor/clear-model/<int:algo_id>', v10.clear_model),
    path('algo_rf_regressor/predict', v10.predict),
    # algorithm: SVM classifier
    path('algo_svm_classifier/add', v11.add_svm_classifier),
    path('algo_svm_classifier/<int:algo_id>', v11.view_svm_classifier),
    path('algo_svm_classifier/import', v11.import_data),
    path('algo_svm_classifier/variables', v11.set_variables),
    path('algo_svm_classifier/clear-variables/<int:algo_id>', v11.clear_variables),
    path('algo_svm_classifier/train-model', v11.train_model),
    path('algo_svm_classifier/clear-model/<int:algo_id>', v11.clear_model),
    path('algo_svm_classifier/predict', v11.predict),
    # algorithm: SVM regressor
    path('algo_svm_regressor/add', v12.add_svm_regressor),
    path('algo_svm_regressor/<int:algo_id>', v12.view_svm_regressor),
    path('algo_svm_regressor/import', v12.import_data),
    path('algo_svm_regressor/variables', v12.set_variables),
    path('algo_svm_regressor/clear-variables/<int:algo_id>', v12.clear_variables),
    path('algo_svm_regressor/train-model', v12.train_model),
    path('algo_svm_regressor/clear-model/<int:algo_id>', v12.clear_model),
    path('algo_svm_regressor/predict', v12.predict),
    # algorithm: One-class SVM
    path('algo_one_class_svm/add', v13.add_one_class_svm),
    path('algo_one_class_svm/<int:algo_id>', v13.view_one_class_svm),
    path('algo_one_class_svm/import', v13.import_data),
    path('algo_one_class_svm/variables', v13.set_variables),
    path('algo_one_class_svm/clear-variables/<int:algo_id>', v13.clear_variables),
    path('algo_one_class_svm/train-model', v13.train_model),
    path('algo_one_class_svm/clear-model/<int:algo_id>', v13.clear_model),
    path('algo_one_class_svm/predict', v13.predict),
    # algorithm: logistic regression
    path('algo_logistic_regression/add', v14.add_logistic_regression),
    path('algo_logistic_regression/<int:algo_id>', v14.view_logistic_regression),
    path('algo_logistic_regression/import', v14.import_data),
    path('algo_logistic_regression/variables', v14.set_variables),
    path('algo_logistic_regression/clear-variables/<int:algo_id>', v14.clear_variables),
    path('algo_logistic_regression/train-model', v14.train_model),
    path('algo_logistic_regression/clear-model/<int:algo_id>', v14.clear_model),
    path('algo_logistic_regression/predict', v14.predict),
    # algorithm: elastic net
    path('algo_elastic_net/add', v15.add_elastic_net),
    path('algo_elastic_net/<int:algo_id>', v15.view_elastic_net),
    path('algo_elastic_net/import', v15.import_data),
    path('algo_elastic_net/variables', v15.set_variables),
    path('algo_elastic_net/clear-variables/<int:algo_id>', v15.clear_variables),
    path('algo_elastic_net/train-model', v15.train_model),
    path('algo_elastic_net/clear-model/<int:algo_id>', v15.clear_model),
    path('algo_elastic_net/predict', v15.predict),
    # algorithm: DBSCAN cluster
    path('algo_dbscan/add', v16.add_dbscan),
    path('algo_dbscan/<int:algo_id>', v16.view_dbscan),
    path('algo_dbscan/import', v16.import_data),
    path('algo_dbscan/variables', v16.set_variables),
    path('algo_dbscan/clear-variables/<int:algo_id>', v16.clear_variables),
    path('algo_dbscan/train-model', v16.train_model),
    path('algo_dbscan/clear-model/<int:algo_id>', v16.clear_model),
    # algorithm: K-means cluster (template for unsupervised learning)
    path('algo_kmeans/add', v17.add_k_means),
    path('algo_kmeans/<int:algo_id>', v17.view_k_means),
    path('algo_kmeans/import', v17.import_data),
    path('algo_kmeans/variables', v17.set_variables),
    path('algo_kmeans/clear-variables/<int:algo_id>', v17.clear_variables),
    path('algo_kmeans/train-model', v17.train_model),
    path('algo_kmeans/clear-model/<int:algo_id>', v17.clear_model),
    path('algo_kmeans/predict', v17.predict),
    # algorithm: PCA
    path('algo_pca/add', v18.add_pca),
    path('algo_pca/<int:algo_id>', v18.view_pca),
    path('algo_pca/import', v18.import_data),
    path('algo_pca/variables', v18.set_variables),
    path('algo_pca/clear-variables/<int:algo_id>', v18.clear_variables),
    path('algo_pca/train-model', v18.train_model),
    path('algo_pca/clear-model/<int:algo_id>', v18.clear_model),
    path('algo_pca/predict', v18.predict),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
