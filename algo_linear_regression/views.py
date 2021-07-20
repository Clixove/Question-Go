import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as linear_regression
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(LinearRegression.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = LinearRegression.objects.get(id=algo_id)


class Note(PublicAlgorithm):
    experiment_note = forms.CharField(widget=forms.Textarea({"class": "form-control"}), required=False, label="")

    def load_note(self, content):
        self.fields['experiment_note'].initial = content


class SearchFile(PublicAlgorithm):
    search_file = forms.CharField(max_length=64, widget=forms.TextInput({"class": "form-control"}), label="")
    data_format = forms.ChoiceField(choices=[(1, "Spreadsheet [*.xlsx]"), (2, "Binary [*.pkl]")],
                                    widget=forms.Select({"class": "form-select"}))


class SelectFile(PublicAlgorithm):
    paper = forms.ModelChoiceField(Paper.objects.none(), widget=forms.Select({"class": "form-select"}),
                                   label="Searching Result", empty_label=None,
                                   help_text="Choose one from searching results.")

    def search_paper(self, user, search_query, role):
        paper_queryset = Paper.objects.filter(name__contains=search_query, user=user, role=role)
        self.fields['paper'].queryset = paper_queryset
        self.fields['paper'].initial = paper_queryset.first()

    def ownership_paper(self, user):
        self.fields['paper'].queryset = Paper.objects.filter(Q(role=1) | Q(role=2), user=user)


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
        help_text="From 1 to 9999999, leave blank if not purpose to fix."
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
    new_step = Step(task=opened_task, name="Linear Regression", view_link=f"/algo_linear_regression/{new_algorithm.id}")
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
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    search_file = SearchFile()
    search_file.link_to_algorithm(algo_id)
    note = Note()
    note.link_to_algorithm(algo_id)
    note.load_note(algorithm_.note)
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
        "algo_lr": algorithm_, "notepad": note, "search_file": search_file, "search_result_empty": SelectFile(),
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var, "train_config": train_config,
        "regression_line_form": rlv,
    }
    # ---------- Load Model Evaluation START ----------
    if algorithm_.evaluate:
        with open(algorithm_.evaluate.file.path, "rb") as f:
            reports = pickle.load(f)
        context['mode'] = reports['mode']
        for table in ['coefficients', 'coefficients_dev', 'significances', 'errors']:
            context[table] = reports[table].to_html(
            classes="table table-sm table-bordered", justify='left', bold_rows=False)
    # ---------- Load Model Evaluation END   ----------
    return render(req, "algo_linear_regression/main.html", context)


@permission_required("algo_linear_regression.change_linearregression")
@csrf_exempt
@require_POST
def change_note(req):
    note = Note(req.POST)
    if not note.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = note.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    algorithm_.note = note.cleaned_data['experiment_note']
    algorithm_.save()
    context = {"color": "success", "content": "Note is changed successfully."}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("library.view_paper")
@csrf_exempt
@require_POST
def search_data(req):
    search_file = SearchFile(req.POST)
    if not search_file.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = search_file.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    select_file = SelectFile()
    select_file.search_paper(req.user, search_file.cleaned_data['search_file'], search_file.cleaned_data['data_format'])
    select_file.link_to_algorithm(algorithm_.id)
    return HttpResponse(select_file.as_p())


@permission_required("algo_linear_regression.change_linearregression")
@csrf_exempt
@require_POST
def use_data(req):
    select_file = SelectFile(req.POST)
    select_file.ownership_paper(req.user)
    if not select_file.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = select_file.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.step.status = 2
    algorithm_.step.save()
    paper = select_file.cleaned_data['paper']
    try:
        # ---------- Asynchronous Algorithm START ----------
        if paper.role == 1:
            table = pd.read_excel(paper.file.path)
        else:  # paper.role == 2
            with open(paper.file.path, "rb") as f:
                table = pickle.load(f)
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2, name=f"Linear Regression #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"lr_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        for column in Column.objects.filter(algorithm=algorithm_):
            column.delete()
        for col in table.targeted_columns:
            new_column = Column(algorithm=algorithm_, name=col)
            new_column.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_linear_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_linear_regression.change_linearregression",
                     login_url="/task/retrieve?message=You don't have access change algorithms.&color=danger")
def confirm_error(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.error_message = ""
    algorithm_.save()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")


@permission_required("algo_linear_regression.change_linearregression",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_data(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.dataframe.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")


@permission_required("algo_linear_regression.change_linearregression")
@csrf_exempt
@require_POST
def set_variables(req):
    variable_picker = VariablePicker(req.POST)
    if not variable_picker.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = variable_picker.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    variable_picker.load_choices(algorithm_)
    if not variable_picker.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
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
    if not algorithm_.open_permission(req.user):
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
    if not train.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = train.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.step.status = 2
    algorithm_.step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        assert algorithm_.dataframe and Column.objects.filter(algorithm=algorithm_, x_column=True).count() > 0 \
               and Column.objects.filter(algorithm=algorithm_, y_column=True).count() == 1
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = [Column.objects.filter(algorithm=algorithm_, y_column=True).first().name]
        mode = train.cleaned_data['running_mode']
        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            x, y = dataframe[x_col].values, dataframe[y_col].values
            dataset, lr_models = [], []
            coefficients, significance, errors = [], [], []
            for train_index, valid_index in k_fold.split(x):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                x_train = linear_regression.add_constant(x_train)
                x_valid = linear_regression.add_constant(x_valid)
                lr_model = linear_regression.OLS(y_train, x_train).fit()
                coefficient = np.array([lr_model.params, lr_model.bse, lr_model.tvalues, lr_model.pvalues]).T
                sig = np.array([lr_model.fvalue, lr_model.f_pvalue, lr_model.rsquared, lr_model.rsquared_adj,
                                lr_model.ssr, lr_model.ess, lr_model.llf])
                y_valid_hat = lr_model.predict(x_valid)
                error = [mean_absolute_error(y_valid, y_valid_hat), mean_squared_error(y_valid, y_valid_hat)]

                lr_models.append(lr_model)
                dataset.append([x_train, x_valid, y_train, y_valid])
                coefficients.append(coefficient)
                significance.append(sig)
                errors.append(error)

            coefficients_avg = np.mean(coefficients, axis=0)
            significance = np.array(significance).T
            errors = np.array(errors).T
            parameters = {
                "coefficients": pd.DataFrame(
                    data=coefficients_avg,
                    columns=['Coefficient', 'Std. Error', 't-Statistic', 'Prob.'],
                    index=["Constant"] + x_col
                ),
                "coefficients_dev": pd.DataFrame(
                    data=np.max([np.max(coefficients, axis=0) - coefficients_avg,
                                 coefficients_avg - np.min(coefficients, axis=0)], axis=0),
                    columns=['Coefficient ±', 'Std. Error ±', 't-Statistic ±', 'Prob. ±'],
                    index=["Constant"] + x_col
                ),
                "significances": pd.DataFrame(
                    data=significance, columns=[f"Fold {i}" for i in range(1, 6)],
                    index=['F-statistic', 'p-value of F-statistic', 'R-squared', 'Adjusted R-squared',
                           'SSR', 'SSE', 'Log likelihood']
                ),
                "mode": mode,
                "errors": pd.DataFrame(
                    data=errors, columns=[f"Fold {i}" for i in range(1, 6)],
                    index=["MAE", "MSE"]
                ),
            }

        elif mode == "split":
            x_train, x_valid, y_train, y_valid = train_test_split(
                dataframe[x_col].values, dataframe[y_col].values, train_size=0.8,
                random_state=train.cleaned_data['random_seed'])
            dataset = [x_train, x_valid, y_train, y_valid]
            x_train = linear_regression.add_constant(x_train)
            x_valid = linear_regression.add_constant(x_valid)
            lr_models = lr_model = linear_regression.OLS(y_train, x_train).fit()
            coefficient = np.array([lr_model.params, lr_model.bse, lr_model.tvalues, lr_model.pvalues]).T
            y_valid_hat = lr_model.predict(x_valid)

            parameters = {
                "coefficients": pd.DataFrame(
                    data=coefficient, columns=['Coefficient', 'Std. Error', 't-Statistic', 'Prob.'],
                    index=["Constant"] + x_col
                ),
                "coefficients_dev": pd.DataFrame(),
                "significances": pd.DataFrame(
                    data=[[lr_model.fvalue], [lr_model.f_pvalue], [lr_model.rsquared], [lr_model.rsquared_adj],
                          [lr_model.ssr], [lr_model.ess], [lr_model.llf]],
                    index=['F-statistic', 'p-value of F-statistic', 'R-squared', 'Adjusted R-squared',
                           'SSR', 'SSE', 'Log likelihood'],
                    columns=["Value"]
                ),
                "mode": mode,
                "errors": pd.DataFrame(
                    data=[mean_absolute_error(y_valid, y_valid_hat), mean_squared_error(y_valid, y_valid_hat)],
                    columns=["Value"], index=["MAE", "MSE"]
                ),
            }

        else:  # mode == "full_train"
            x_train, y_train = dataframe[x_col].values, dataframe[y_col].values
            dataset = [x_train, y_train]
            x_train = linear_regression.add_constant(x_train)
            lr_models = lr_model = linear_regression.OLS(y_train, x_train).fit()
            coefficient = np.array([lr_model.params, lr_model.bse, lr_model.tvalues, lr_model.pvalues]).T

            parameters = {
                "coefficients": pd.DataFrame(
                    data=coefficient, columns=['Coefficient', 'Std. Error', 't-Statistic', 'Prob.'],
                    index=["Constant"] + x_col
                ),
                "coefficients_dev": pd.DataFrame(),
                "significances": pd.DataFrame(
                    data=[[lr_model.fvalue], [lr_model.f_pvalue], [lr_model.rsquared], [lr_model.rsquared_adj],
                          [lr_model.ssr], [lr_model.ess], [lr_model.llf]],
                    index=['F-statistic', 'p-value of F-statistic', 'R-squared', 'Adjusted R-squared',
                           'SSR', 'SSE', 'Log likelihood'],
                    columns=["Value"]
                ),
                "mode": mode,
                "errors": pd.DataFrame(),
            }
        # zip parameters
        parameters_bin = ContentFile(pickle.dumps(parameters))
        parameters_paper = Paper(user=req.user, role=2, name=f"Linear Regression #{algorithm_.id} Parameters")
        parameters_paper.file.save(f"lr_{algorithm_.id}_paras_{mode}.pkl", parameters_bin)
        algorithm_.evaluate = parameters_paper

        # zip models
        models_bin = ContentFile(pickle.dumps(lr_models))
        models_paper = Paper(user=req.user, role=3, name=f"Linear Regression #{algorithm_.id} Model")
        models_paper.file.save(f"lr_{algorithm_.id}_model_{mode}.pkl", models_bin)
        algorithm_.model = models_paper

        # zip dataset
        dataset_bin = ContentFile(pickle.dumps(dataset))
        dataset_paper = Paper(user=req.user, role=2, name=f"Linear Regression #{algorithm_.id} Matrix")
        dataset_paper.file.save(f"lr_{algorithm_.id}_matrix_{mode}.pkl", dataset_bin)
        dataset_paper.save()
        algorithm_.matrix = dataset_paper
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_linear_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
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
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model.delete()
    algorithm_.matrix.delete()
    algorithm_.evaluate.delete()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")


@permission_required("algo_linear_regression.view_linearregression")
@csrf_exempt
@require_POST
def regression_line(req):
    rlv = RegressionLineVariable(req.POST)
    if not rlv.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = rlv.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    rlv.load_choices(algorithm_)
    if not rlv.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        x_col = rlv.cleaned_data['variable'].name
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        other_variables = [x.name for x in
                           Column.objects.filter(algorithm=algorithm_, x_column=True).exclude(name=x_col)]
        with open(algorithm_.evaluate.file.path, "rb") as f:
            evaluation = pickle.load(f)
        other_slope = {j: evaluation['coefficients'].loc[j, 'Coefficient'] for j in other_variables}
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        f, fig = io.StringIO(), plt.figure()
        x, y = dataframe[x_col].values, dataframe[y_col].values
        slope = evaluation['coefficients'].loc[x_col, 'Coefficient']
        intercept = np.sum([dataframe[j].mean(axis=0) * other_slope[j] for j in other_slope]) + \
            evaluation['coefficients'].loc['Constant', 'Coefficient']
        plt.plot([x.min(), x.max()], [x.min() * slope + intercept, x.max() * slope + intercept], "r")
        sampling = np.random.choice(x.shape[0], 100)
        plt.scatter(x[sampling], y[sampling])
        plt.xlabel(x_col), plt.ylabel(y_col)
        plt.legend(["regression line", "sampling of data"])
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
    select_file = SelectFile(req.POST)
    select_file.ownership_paper(req.user)
    if not select_file.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = select_file.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------

    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        with open(algorithm_.evaluate.file.path, "rb") as f:
            evaluation = pickle.load(f)
        with open(algorithm_.model.file.path, "rb") as f:
            lr_model = pickle.load(f)
        paper = select_file.cleaned_data['paper']
        if paper.role == 1:
            table = pd.read_excel(paper.file.path)
        else:
            with open(paper.file.path, "rb") as f:
                table = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        mode = evaluation['mode']
        x = linear_regression.add_constant(table[x_col].values)
        if mode == '5_fold':
            y_hat = np.empty(shape=(table.shape[0], 5))
            for i in range(5):
                y_hat[:, i] = lr_model[i].predict(x)
            table[y_col] = np.nanmean(y_hat, axis=1)
        else:
            table[y_col] = lr_model.predict(x)
        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            table.to_excel(f, index=False)
    except Exception as e:
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"Linear Regression #{algorithm_.id} Predict")
    new_paper.file.save(f"lr_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    algorithm_.predict = new_paper
    algorithm_.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_linear_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)


@permission_required("algo_linear_regression.change_linearregression",
                         login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_predict(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = LinearRegression.objects.get(id=algo_id)
    except LinearRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.predict.delete()
    algorithm_.save()
    return redirect(f"/algo_linear_regression/{algorithm_.id}")
