import io
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as linear_regression
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(LinearRegression.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = LinearRegression.objects.get(id=algo_id)


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


class RegressionLineVariable(PublicAlgorithm):
    variable = forms.ModelChoiceField(
        Column.objects.all(), widget=forms.Select({"class": "form-select"}), initial=""
    )

    def load_choices(self, algorithm):
        variable = Column.objects.filter(algorithm=algorithm, x_column=True)
        self.fields['variable'].queryset = variable
        self.fields['variable'].initial = variable.first()


@permission_required("algo_linear_regression.add_linearregression",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_lr(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = LinearRegression()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Linear Regression", view_link=f"/algo_linear_regression/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_linear_regression.view_linearregression",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_lr(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
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
    rlv = RegressionLineVariable()
    rlv.link_to_algorithm(algo_id)
    rlv.load_choices(algorithm_)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/algo_linear_regression/import',
        "predict_data_target": '/algo_linear_regression/predict',
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var, "train_config": train_config,
        "regression_line_form": rlv,
    }
    try:
        context['coefficients'] = json.loads(algorithm_.coefficients)
        context['significances'] = json.loads(algorithm_.significances)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_linear_regression/main.html", context)


@permission_required("algo_linear_regression.change_linearregression")
@csrf_exempt
@require_POST
def import_data(req):
    # ---------- Import Data Tool START ----------
    flag, content = task_manager.views.use_data(req, train=False)
    if flag:
        context = {'color': 'danger', 'content': 'Submission is not valid.'}
        return render(req, 'task_manager/hint_widget.html', context)
    step, table = content
    # ---------- Import Data Tool End   ----------
    algorithm_ = LinearRegression.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"Linear Regression #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"linear_regression_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/algo_linear_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_linear_regression.change_linearregression")
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
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_linear_regression.change_linearregression",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = column.y_column = False
        column.save()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")


@permission_required("algo_linear_regression.change_linearregression")
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

        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            models, coefficients, significances, errors = [], [], [], []

            for train_index, valid_index in k_fold.split(x):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                x_train = linear_regression.add_constant(x_train)
                x_valid = linear_regression.add_constant(x_valid)
                mdl = linear_regression.OLS(y_train, x_train).fit()
                y_valid_hat = mdl.predict(x_valid)

                coef = {}
                names = ['Constant'] + x_col
                for name, i in zip(names, range(len(names))):
                    coef[name] = {
                        'coef': mdl.params[i], 'std_error': mdl.bse[i], 't': mdl.tvalues[i], 'p': mdl.pvalues[i]
                    }
                sig = {
                    'f': mdl.fvalue, 'p': mdl.f_pvalue, 'R2': mdl.rsquared, 'R2_adj': mdl.rsquared_adj,
                    'SSR': mdl.ssr, 'SSE': mdl.ess, 'log_likelihood_f': mdl.llf,
                    'MAE': mean_absolute_error(y_valid, y_valid_hat), 'MSE': mean_squared_error(y_valid, y_valid_hat)
                }
                models.append(mdl)
                coefficients.append(coef)
                significances.append(sig)
            models_bin = ContentFile(pickle.dumps(models))
            algorithm_.coefficients = json.dumps(coefficients, ensure_ascii=False)
            algorithm_.significances = json.dumps(significances, ensure_ascii=False)

        elif mode == "split":
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, shuffle=True,
                                                                  random_state=train.cleaned_data['random_seed'])
            x_train = linear_regression.add_constant(x_train)
            x_valid = linear_regression.add_constant(x_valid)
            mdl = linear_regression.OLS(y_train, x_train).fit()
            y_valid_hat = mdl.predict(x_valid)

            coef = {}
            names = ['Constant'] + x_col
            for x, i in zip(names, range(len(names))):
                coef[x] = {
                    'coef': mdl.params[i], 'std_error': mdl.bse[i], 't': mdl.tvalues[i], 'p': mdl.pvalues[i]
                }
            sig = {
                'f': mdl.fvalue, 'p': mdl.f_pvalue, 'R2': mdl.rsquared, 'R2_adj': mdl.rsquared_adj,
                'SSR': mdl.ssr, 'SSE': mdl.ess, 'log_likelihood_f': mdl.llf,
                'MAE': mean_absolute_error(y_valid, y_valid_hat), 'MSE': mean_squared_error(y_valid, y_valid_hat)
            }
            models_bin = ContentFile(pickle.dumps(mdl))
            algorithm_.coefficients = json.dumps(coef, ensure_ascii=False)
            algorithm_.significances = json.dumps(sig, ensure_ascii=False)

        else:  # mode == "full_train"
            x = linear_regression.add_constant(x)
            mdl = linear_regression.OLS(y, x).fit()
            coef = {}
            names = ['Constant'] + x_col
            for x, i in zip(names, range(len(names))):
                coef[x] = {
                    'coef': mdl.params[i], 'std_error': mdl.bse[i], 't': mdl.tvalues[i], 'p': mdl.pvalues[i]
                }
            sig = {
                'f': mdl.fvalue, 'p': mdl.f_pvalue, 'R2': mdl.rsquared, 'R2_adj': mdl.rsquared_adj,
                'SSR': mdl.ssr, 'SSE': mdl.ess, 'log_likelihood_f': mdl.llf,
                'MAE': np.nan, 'MSE': np.nan
            }
            models_bin = ContentFile(pickle.dumps(mdl))
            algorithm_.coefficients = json.dumps(coef, ensure_ascii=False)
            algorithm_.significances = json.dumps(sig, ensure_ascii=False)

        models_paper = Paper(user=req.user, role=3, name=f"Linear Regression #{algorithm_.id} Model")
        models_paper.file.save(f"linear_regression_{algorithm_.id}_model_{mode}.pkl", models_bin)
        models_paper.save()
        algorithm_.model = models_paper
        algorithm_.mode = mode
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_linear_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_linear_regression.change_linearregression",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.mode = str()
    algorithm_.significances = str()
    algorithm_.coefficients = str()
    algorithm_.model = None
    algorithm_.save()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")


@permission_required("algo_linear_regression.view_linearregression")
@csrf_exempt
@require_POST
def regression_line(req):
    rlv = RegressionLineVariable(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = rlv.is_valid()
    algorithm_ = rlv.cleaned_data['algorithm']
    v2 = algorithm_.step.open_permission(req.user)
    rlv.load_choices(algorithm_)
    v3 = rlv.is_valid()
    if not (v1 and v2 and v3):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        x_col = rlv.cleaned_data['variable'].name
        other_cols = [j.name for j in Column.objects.filter(algorithm=algorithm_, x_column=True).exclude(name=x_col)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        f, fig = io.StringIO(), plt.figure()
        x, y = dataframe[x_col].values, dataframe[y_col].values
        coefficients = json.loads(algorithm_.coefficients)
        if algorithm_.mode == '5_fold':
            coef = [coef_each_fold[x_col]['coef'] for coef_each_fold in coefficients]
            intercept = [np.sum([dataframe[j].mean(axis=0) * coef_each_fold[j]['coef'] for j in other_cols]) +
                         coef_each_fold['Constant']['coef'] for coef_each_fold in coefficients]
            slope = np.mean(coef)
            intercept = np.mean(intercept)
        else:
            slope = coefficients[x_col]['coef']
            intercept = np.sum([dataframe[j].mean(axis=0) * coefficients[j]['coef'] for j in other_cols]) + \
                coefficients['Constant']['coef']
        plt.plot([x.min(), x.max()], [x.min() * slope + intercept, x.max() * slope + intercept], "r")
        sampling = np.random.choice(x.shape[0], 100)
        plt.scatter(x[sampling], y[sampling])
        plt.xlabel(x_col), plt.ylabel(y_col)
        plt.legend(["regression line", "100 sampled data"])
        fig.savefig(f, format='svg')
        plt.close(fig)
    except Exception as e:
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    return HttpResponse(f.getvalue())


@permission_required("algo_linear_regression.view_linearregression")
@csrf_exempt
@require_POST
def predict(req):
    # ---------- Import Data Tool START ----------
    flag, content = task_manager.views.use_data(req, train=False)
    if flag:
        context = {'color': 'danger', 'content': 'Submission is not valid.'}
        return render(req, 'task_manager/hint_widget.html', context)
    step, table = content
    # ---------- Import Data Tool End   ----------
    algorithm_ = LinearRegression.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        x = linear_regression.add_constant(table[x_col].values)
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
    new_paper = Paper(user=req.user, role=4, name=f"Linear Regression #{algorithm_.id} Predict")
    new_paper.file.save(f"linear_regression_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
