import io
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold

import task_manager.views
from bayes_opt import BayesianOptimization
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(BayesLogisticRegression.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = BayesLogisticRegression.objects.get(id=algo_id)


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
    bayes_init_try_times = forms.IntegerField(
        min_value=16, widget=forms.NumberInput({'class': 'form-control'}), required=None,
        help_text='At least 16. How many steps of random exploration you want to perform. '
                  'Random exploration can help by diversifying the exploration space.'
    )
    bayes_iteration_times = forms.IntegerField(
        min_value=16, max_value=100, widget=forms.NumberInput({'class': 'form-control'}), required=None,
        help_text='From 16 to 100. How many steps of bayesian optimization you want to perform. The more steps the '
                  'more likely to find a good maximum you are.'
    )
    regularization = forms.ChoiceField(
        widget=forms.Select({'class': 'form-control form-select'}),
        choices=[('none', 'No'), ('l1', 'L1'), ('l2', 'L2'), ('elasticnet', 'Mixing L1 and L2')],
        initial='none',
    )
    min_ln_c = forms.FloatField(
        widget=forms.NumberInput({'class': 'form-control'}), required=False,
        help_text='The logarithmic value of regularization strength, inversely proportional to the strength of '
                  'the regularization.',
    )
    max_ln_c = forms.FloatField(
        widget=forms.NumberInput({'class': 'form-control'}), required=False,
        help_text='The logarithmic value of regularization parameter, inversely proportional to the strength of '
                  'the regularization.',
    )


@permission_required(
    'algo_logistic_regression.add_bayeslogisticregression',
    login_url='/task/retrieve?message=You don\'t have access to this algorithm.&color=danger'
)
def add_logistic_regression(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = BayesLogisticRegression()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Logistic Regression", view_link=f"/algo_logistic_regression/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_logistic_regression.view_bayeslogisticregression",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_logistic_regression(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesLogisticRegression.objects.get(id=algo_id)
    except BayesLogisticRegression.DoesNotExist:
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
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/algo_logistic_regression/import',
        "predict_data_target": '/algo_logistic_regression/predict',
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var,
        "train_config": train_config,
    }
    try:
        context['bayes_history'] = json.loads(algorithm_.training_history)
        context['h_para'] = json.loads(algorithm_.hyper_parameters)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_logistic_regression/main.html", context)


@permission_required("algo_logistic_regression.change_bayeslogisticregression")
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
    algorithm_ = BayesLogisticRegression.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"Logistic Regression #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"lgr_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_logistic_regression.change_bayeslogisticregression")
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
               "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_logistic_regression.change_bayeslogisticregression",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesLogisticRegression.objects.get(id=algo_id)
    except BayesLogisticRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = column.y_column = False
        column.save()
    return redirect(f"/algo_logistic_regression/{algorithm_.id}")


def one_hot(labels):
    u, i = np.unique(labels, return_inverse=True)
    labels_1h = np.eye(u.shape[0])[i]
    class_dict_ = dict(zip(u, range(len(u))))
    return class_dict_, labels_1h


@permission_required("algo_logistic_regression.change_bayeslogisticregression")
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

    if train.cleaned_data['regularization'] != 'none':
        if not (
            train.cleaned_data['bayes_init_try_times'] and train.cleaned_data['bayes_iteration_times'] and
            train.cleaned_data['min_ln_c'] and train.cleaned_data['max_ln_c'] and
            train.cleaned_data['min_ln_c'] < train.cleaned_data['max_ln_c']
        ):
            context = {"color": "danger", "content": "Submission is not valid."}
            return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        mode = train.cleaned_data['running_mode']
        x, y = dataframe[x_col].values, dataframe[y_col].values

        class_dict, y_1h = one_hot(y)
        fpr_poly_ = np.linspace(0, 1, 200)
        hyper_parameters = {
            'c': (train.cleaned_data['min_ln_c'], train.cleaned_data['max_ln_c']),
            'l1_ratio': (0, 1),
        }
        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            models_, histories, auc_s = [], [], []
            auc = {name: np.zeros(5).tolist() for name in class_dict.keys()}
            tpr_poly_ = np.zeros((y_1h.shape[1], 5, 200))
            hyper_parameters_list = []
            for (train_index, valid_index), k in zip(k_fold.split(x), range(5)):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                y_1h_train, y_1h_valid = y_1h[train_index], y_1h[valid_index]

                if train.cleaned_data['regularization'] != 'none':
                    def bayes_lgr_5_fold(c, l1_ratio):
                        lgr = LogisticRegression(
                            penalty=train.cleaned_data['regularization'],
                            solver='saga', C=np.exp(c), l1_ratio=l1_ratio,
                            random_state=train.cleaned_data['random_seed'],
                        )
                        lgr.fit(x_train, y_train)
                        y_train_hat = lgr.predict_proba(x_train)
                        auc_in_bayes = np.mean([roc_auc_score(y_1h_train[:, i], y_train_hat[:, i])
                                                for i in range(y_1h.shape[1])])
                        return auc_in_bayes
                    optimizer = BayesianOptimization(f=bayes_lgr_5_fold, pbounds=hyper_parameters,
                                                     random_state=train.cleaned_data['random_seed'])
                    optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                                       n_iter=train.cleaned_data['bayes_iteration_times'])
                    history = {i: res for i, res in enumerate(optimizer.res)}
                    histories.append(history)
                    mdl = LogisticRegression(
                        penalty=train.cleaned_data['regularization'],
                        solver='saga', C=np.exp(optimizer.max['params']['c']),
                        l1_ratio=optimizer.max['params']['l1_ratio'],
                        random_state=train.cleaned_data['random_seed'],
                    )
                else:
                    mdl = LogisticRegression(penalty='none', random_state=train.cleaned_data['random_seed'])
                mdl.fit(x_train, y_train)
                models_.append(mdl)
                y_valid_hat = mdl.predict_proba(x_valid)
                for name, i in class_dict.items():
                    auc[name][k] = roc_auc_score(y_1h_valid[:, i], y_valid_hat[:, i])
                    fpr, tpr, _ = roc_curve(y_1h_valid[:, i], y_valid_hat[:, i])
                    tpr_poly_[i, k, :] = np.interp(fpr_poly_, fpr, tpr)
                hyper_parameters_list.append({
                    'penalty': mdl.penalty, 'solver': mdl.solver, 'c': mdl.C,
                    'l1_ratio': mdl.l1_ratio
                })
            f, fig = io.StringIO(), plt.figure()
            for name, i in class_dict.items():
                mean_roc = np.mean(tpr_poly_, axis=1)[i]
                mean_auc = np.mean(auc[name])
                range_auc = np.maximum(np.max(auc[name]) - mean_auc, mean_auc - np.min(auc[name]))
                plt.plot(fpr_poly_, mean_roc, label=f'{name} (AUC = {mean_auc.round(3)} ± {range_auc.round(3)})')

            intermediate_paper_handle = ContentFile(pickle.dumps(models_))
            new_paper = Paper(user=req.user, role=3, name=f'Logistic Regression #{algorithm_.id} Model')
            new_paper.file.save(f'lgr_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
            algorithm_.training_history = json.dumps(histories, ensure_ascii=False)
            algorithm_.auc = json.dumps(auc, ensure_ascii=False)

            plt.plot([0, 1], [0, 1], linestyle='--', lw=1.25, color='b', label='Chance')
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend(loc=4)
            fig.savefig(f, format='svg')
            plt.close(fig)
            algorithm_.roc_curve = f.getvalue()
            algorithm_.hyper_parameters = json.dumps(hyper_parameters_list, ensure_ascii=False)

        elif mode == 'split':
            x_train, x_valid, y_train, y_valid, y_1h_train, y_1h_valid = train_test_split(
                x, y, y_1h, train_size=0.8, shuffle=True, random_state=train.cleaned_data['random_seed'])
            if train.cleaned_data['regularization'] != 'none':
                def bayes_lgr_split(c, l1_ratio):
                    lgr = LogisticRegression(
                        penalty=train.cleaned_data['regularization'],
                        solver='saga', C=np.exp(c), l1_ratio=l1_ratio,
                        random_state=train.cleaned_data['random_seed'],
                    )
                    lgr.fit(x_train, y_train)
                    y_train_hat = lgr.predict_proba(x_train)
                    auc_in_bayes = np.mean([roc_auc_score(y_1h_train[:, i], y_train_hat[:, i])
                                            for i in range(y_1h.shape[1])])
                    return auc_in_bayes

                optimizer = BayesianOptimization(f=bayes_lgr_split, pbounds=hyper_parameters,
                                                 random_state=train.cleaned_data['random_seed'])
                optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                                   n_iter=train.cleaned_data['bayes_iteration_times'])
                history = {i: res for i, res in enumerate(optimizer.res)}
                algorithm_.training_history = json.dumps(history, ensure_ascii=False)
                mdl = LogisticRegression(
                    penalty=train.cleaned_data['regularization'],
                    solver='saga', C=np.exp(optimizer.max['params']['c']),
                    l1_ratio=optimizer.max['params']['l1_ratio'],
                    random_state=train.cleaned_data['random_seed'],
                )
            else:
                mdl = LogisticRegression(penalty='none', random_state=train.cleaned_data['random_seed'])
            mdl.fit(x_train, y_train)
            y_valid_hat = mdl.predict_proba(x_valid)
            hyper_parameters = {
                'penalty': mdl.penalty, 'solver': mdl.solver, 'c': mdl.C,
                'l1_ratio': mdl.l1_ratio
            }
            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)
            auc = {}
            f, fig = io.StringIO(), plt.figure()

            for name, i in class_dict.items():
                auc[name] = roc_auc_score(y_1h_valid[:, i], y_valid_hat[:, i])
                fpr, tpr, _ = roc_curve(y_1h_valid[:, i], y_valid_hat[:, i])
                tpr_poly_ = np.interp(fpr_poly_, fpr, tpr)
                plt.plot(fpr_poly_, tpr_poly_, label=f'{name} (AUC = {auc[name].__round__(3)})')

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Logistic Regression #{algorithm_.id} Model')
            new_paper.file.save(f'lgr_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
            algorithm_.auc = json.dumps(auc, ensure_ascii=False)

            plt.plot([0, 1], [0, 1], linestyle='--', lw=1.25, color='b', label='Chance')
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend(loc=4)
            fig.savefig(f, format='svg')
            plt.close(fig)
            algorithm_.roc_curve = f.getvalue()
        else:
            if train.cleaned_data['regularization'] != 'none':
                def bayes_lgr_full_train(c, l1_ratio):
                    lgr = LogisticRegression(
                        penalty=train.cleaned_data['regularization'],
                        solver='saga', C=np.exp(c), l1_ratio=l1_ratio,
                        random_state=train.cleaned_data['random_seed'],
                    )
                    lgr.fit(x, y)
                    y_hat = lgr.predict_proba(x)
                    auc_in_bayes = np.mean([roc_auc_score(y_1h[:, i], y_hat[:, i])
                                            for i in range(y_1h.shape[1])])
                    return auc_in_bayes

                optimizer = BayesianOptimization(f=bayes_lgr_full_train, pbounds=hyper_parameters,
                                                 random_state=train.cleaned_data['random_seed'])
                optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                                   n_iter=train.cleaned_data['bayes_iteration_times'])
                history = {i: res for i, res in enumerate(optimizer.res)}
                algorithm_.training_history = json.dumps(history, ensure_ascii=False)
                mdl = LogisticRegression(
                    penalty=train.cleaned_data['regularization'],
                    solver='saga', C=np.exp(optimizer.max['params']['c']),
                    l1_ratio=optimizer.max['params']['l1_ratio'],
                    random_state=train.cleaned_data['random_seed'],
                )
            else:
                mdl = LogisticRegression(penalty='none', random_state=train.cleaned_data['random_seed'])
            mdl.fit(x, y)
            hyper_parameters = {
                'penalty': mdl.penalty, 'solver': mdl.solver, 'c': mdl.C,
                'l1_ratio': mdl.l1_ratio
            }
            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)
            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Logistic Regression #{algorithm_.id} Model')
            new_paper.file.save(f'lgr_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper

        algorithm_.class_dict = json.dumps(class_dict, ensure_ascii=False)
        algorithm_.mode = mode
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_logistic_regression.change_bayeslogisticregression",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesLogisticRegression.objects.get(id=algo_id)
    except BayesLogisticRegression.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model = None
    algorithm_.class_dict = str()
    algorithm_.training_history = str()
    algorithm_.mode = str()
    algorithm_.auc = str()
    algorithm_.hyper_parameters = str()
    algorithm_.roc_curve = str()
    algorithm_.save()
    return redirect(f"/algo_logistic_regression/{algorithm_.id}")


def most_frequent_item(a: np.array):
    a = a.tolist()
    return max(set(a), key=a.count)


@permission_required("algo_logistic_regression.change_bayeslogisticregression",
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
    algorithm_ = BayesLogisticRegression.objects.get(step=step)
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
            y_hat = np.array([model[i].predict(x) for i in range(5)], dtype='object')
            y_hat = np.apply_along_axis(most_frequent_item, axis=0, arr=y_hat)
            table[y_col] = y_hat
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
    new_paper = Paper(user=req.user, role=4, name=f"Logistic Regression #{algorithm_.id} Predict")
    new_paper.file.save(f"lgr_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_logistic_regression/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
