import io
import json
import pickle

import numpy as np
import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(MyElasticNet.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = MyElasticNet.objects.get(id=algo_id)


class VariablePicker(PublicAlgorithm):
    Independent_Variables_X = forms.ModelMultipleChoiceField(
        Column.objects.all(), widget=forms.CheckboxSelectMultiple()
    )
    Dependent_Variable_Y = forms.ModelChoiceField(
        Column.objects.all(), widget=forms.Select({"class": "form-select"}), initial=""
    )

    def load_choices(self, algorithm):
        columns = Column.objects.filter(algorithm=algorithm)
        self.fields['Independent_Variables_X'].queryset = columns
        self.fields['Dependent_Variable_Y'].queryset = columns
        self.fields['Dependent_Variable_Y'].initial = columns.last()


class Train(PublicAlgorithm):
    running_mode = forms.ChoiceField(
        widget=forms.Select({"class": "form-select"}),
        choices=[("5_fold", "5 fold cross validation"),
                 ("split", "Random split to 80% training set, 20% validation set"),
                 ("full_train", "Applying all samples for training")],
    )
    random_seed = forms.IntegerField(
        min_value=1, max_value=9999999, required=False, widget=forms.NumberInput({"class": "form-control"}),
        help_text="Not required. From 1 to 9999999, leave blank if not purpose to fix."
    )
    criterion = forms.ChoiceField(
        choices=[('mae', 'Mean Absolute Error'), ('mse', 'Mean Squared Error')],
        initial='mse',
        help_text='The function to measure the quality of a split.',
        widget=forms.Select({'class': 'form-select'})
    )
    l1 = forms.FloatField(
        min_value=0, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='parameters, referring to OLS loss function'
    )
    l2 = forms.FloatField(
        min_value=0, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='parameters, referring to OLS loss function'
    )


@permission_required("algo_elastic_net.add_myelasticnet",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_elastic_net(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = MyElasticNet()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Elastic Net", view_link=f"/algo_elastic_net/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_elastic_net.view_myelasticnet",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_elastic_net(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyElasticNet.objects.get(id=algo_id)
    except MyElasticNet.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    variable_picker = VariablePicker()
    variable_picker.link_to_algorithm(algorithm_.id)
    variable_picker.load_choices(algorithm_)
    x_var = algorithm_.column_set.filter(x_column=True)
    y_var = algorithm_.column_set.filter(y_column=True)
    train_config = Train()
    train_config.link_to_algorithm(algorithm_.id)
    train_config.fields['l1'].initial = algorithm_.l1
    train_config.fields['l2'].initial = algorithm_.l2
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/algo_elastic_net/import',
        "predict_data_target": '/algo_elastic_net/predict',
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var,
        "train_config": train_config,
    }
    try:
        context['error_measure'] = json.loads(algorithm_.error_measure)
        context['coefficients'] = json.loads(algorithm_.coefficients)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_elastic_net/main.html", context)


@permission_required("algo_elastic_net.change_myelasticnet")
@csrf_exempt
@require_POST
def import_data(req):
    # ---------- Import Data Tool V2 START ----------
    table, step, error_message = task_manager.views.import_training_set_v2(req)
    if table is None:
        context = {'color': 'danger', 'content': error_message}
        return render(req, 'task_manager/hint_widget.html', context)
    # "step.status" has been changed to 2.
    # ---------- Import Data Tool V2 END   ----------
    algorithm_ = MyElasticNet.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"Elastic Net #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"elastic_net_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        algorithm_.save()
        for column in Column.objects.filter(algorithm=algorithm_):
            column.delete()
        for col in table.columns:
            new_column = Column(algorithm=algorithm_, name=col)
            new_column.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_elastic_net/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_elastic_net/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_elastic_net.change_myelasticnet")
@csrf_exempt
@require_POST
def set_variables(req):
    variable_picker = VariablePicker(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = variable_picker.is_valid()
    algorithm_ = variable_picker.cleaned_data['algorithm']
    v2 = algorithm_.step.open_permission(req.user)
    variable_picker.load_choices(algorithm_)
    v3 = variable_picker.is_valid()
    if not (v1 and v2 and v3):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    for column in variable_picker.cleaned_data['Independent_Variables_X']:
        column.x_column = True
        column.save()
    column = variable_picker.cleaned_data['Dependent_Variable_Y']
    column.y_column = True
    column.save()
    context = {"color": "success", "content": "Set variables successfully.",
               "refresh": f"/algo_elastic_net/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_elastic_net.change_myelasticnet",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyElasticNet.objects.get(id=algo_id)
    except MyElasticNet.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = column.y_column = False
        column.save()
    return redirect(f"/algo_elastic_net/{algorithm_.id}")


@permission_required("algo_elastic_net.change_myelasticnet")
@csrf_exempt
@require_POST
def train_model(req):
    train = Train(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = train.is_valid()
    algorithm_ = train.cleaned_data['algorithm']
    step = algorithm_.step
    v2 = step.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    if step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = [Column.objects.filter(algorithm=algorithm_, y_column=True).first().name]
        mode = train.cleaned_data['running_mode']
        x, y = dataframe[x_col].values, dataframe[y_col].values
        func_error = mean_absolute_error if train.cleaned_data['criterion'] == 'mae' else mean_squared_error
        algorithm_.l1, algorithm_.l2 = train.cleaned_data['l1'], train.cleaned_data['l2']
        a = algorithm_.l1 + algorithm_.l2
        b = 0 if a == 0 else algorithm_.l1 / a
        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            models_, coefficients_list = [], []
            error_measure = {'type': train.cleaned_data['criterion'], 'value': []}

            for (train_index, valid_index), k in zip(k_fold.split(x), range(5)):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                mdl = ElasticNet(alpha=a, l1_ratio=b, random_state=train.cleaned_data['random_seed'])
                mdl.fit(x_train, y_train.ravel())
                models_.append(mdl)
                y_valid_hat = mdl.predict(x_valid)
                coefficients_list.append(
                    dict(zip(x_col, mdl.coef_)) | {'intercept': mdl.intercept_}
                )
                error_measure['value'].append(func_error(y_valid, y_valid_hat))
            intermediate_paper_handle = ContentFile(pickle.dumps(models_))
            new_paper = Paper(user=req.user, role=3, name=f'Elastic Net #{algorithm_.id} Model')
            new_paper.file.save(f'elastic_net_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
            algorithm_.coefficients = json.dumps(coefficients_list, ensure_ascii=False)
            algorithm_.error_measure = json.dumps(error_measure, ensure_ascii=False)
        elif mode == 'split':
            x_train, x_valid, y_train, y_valid = train_test_split(
                x, y, train_size=0.8, shuffle=True, random_state=train.cleaned_data['random_seed'])

            mdl = ElasticNet(alpha=a, l1_ratio=b, random_state=train.cleaned_data['random_seed'])
            mdl.fit(x_train, y_train.ravel())
            y_valid_hat = mdl.predict(x_valid)
            coefficients = dict(zip(x_col, mdl.coef_)) | {'intercept': mdl.intercept_}
            algorithm_.coefficients = json.dumps(coefficients, ensure_ascii=False)
            algorithm_.error_measure = json.dumps(
                {'type': train.cleaned_data['criterion'], 'value': func_error(y_valid, y_valid_hat)},
                ensure_ascii=False
            )
            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Elastic Net #{algorithm_.id} Model')
            new_paper.file.save(f'elastic_net_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper

        else:  # mode == "full_train"
            mdl = ElasticNet(alpha=a, l1_ratio=b, random_state=train.cleaned_data['random_seed'])
            mdl.fit(x, y.ravel())
            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Elastic Net #{algorithm_.id} Model')
            new_paper.file.save(f'elastic_net_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
        algorithm_.mode = mode
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_elastic_net/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_elastic_net/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_elastic_net.change_myelasticnet",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyElasticNet.objects.get(id=algo_id)
    except MyElasticNet.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model = None
    algorithm_.training_history = str()
    algorithm_.error_measure = str()
    algorithm_.mode = str()
    algorithm_.hyper_parameters = str()
    algorithm_.feature_importance = str()
    algorithm_.save()
    return redirect(f"/algo_elastic_net/{algorithm_.id}")


@permission_required("algo_elastic_net.change_myelasticnet",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
@csrf_exempt
@require_POST
def predict(req):
    # ---------- Import Data Tool V2 START ----------
    table, step, error_message = task_manager.views.import_predicting_set_v2(req)
    if table is None:
        context = {'color': 'danger', 'content': error_message}
        return render(req, 'task_manager/hint_widget.html', context)
    # "step.status" has been changed to 2.
    # ---------- Import Data Tool V2 END   ----------
    algorithm_ = MyElasticNet.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        x = table[x_col].values
        if algorithm_.mode == '5_fold':
            y_hat = [model[i].predict(x) for i in range(5)]
            table[y_col] = np.nanmean(y_hat, axis=0)
        else:
            table[y_col] = model.predict(x)
        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            table.to_excel(f, index=False)
    except Exception as e:
        step.status = 4
        step.save()
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"Elastic Net #{algorithm_.id} Predict")
    new_paper.file.save(f"rf_regressor_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_elastic_net/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
