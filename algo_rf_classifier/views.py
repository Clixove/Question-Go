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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, train_test_split

import task_manager.views
from bayes_opt import BayesianOptimization
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(BayesRfClassifier.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = BayesRfClassifier.objects.get(id=algo_id)


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
    max_depth_min = forms.IntegerField(
        min_value=1, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='The maximum depth of the tree. (no less than 1)'
    )
    max_depth_max = forms.IntegerField(
        min_value=1, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='The maximum depth of the tree. (no less than 1)'
    )
    max_leaf_nodes_min = forms.IntegerField(
        min_value=2, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='This algorithm grows trees with max_leaf_nodes in best-first fashion. (no less than 2)'
    )
    max_leaf_nodes_max = forms.IntegerField(
        min_value=2, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='This algorithm grows trees with max_leaf_nodes in best-first fashion. (no less than 2)'
    )
    criterion = forms.ChoiceField(
        choices=[('gini', 'Gini impurity'), ('entropy', 'Information gain')],
        initial='gini',
        help_text='The function to measure the quality of a split.',
        widget=forms.Select({'class': 'form-select'})
    )
    n_estimators_min = forms.IntegerField(
        min_value=2, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='The max number of trees in the forest. (no less than 2)'
    )
    n_estimators_max = forms.IntegerField(
        min_value=2, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='The max number of trees in the forest. (no less than 2)'
    )
    bayes_init_try_times = forms.IntegerField(
        min_value=16, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='At least 16. How many steps of random exploration you want to perform. '
                  'Random exploration can help by diversifying the exploration space.'
    )
    bayes_iteration_times = forms.IntegerField(
        min_value=16, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='From 16 to 100. How many steps of bayesian optimization you want to perform. The more steps the '
                  'more likely to find a good maximum you are.'
    )


@permission_required(
    'algo_rf_classifier.add_bayesrfclassifier',
    login_url='/task/retrieve?message=You don\'t have access to this algorithm.&color=danger'
)
def add_rf_classifier(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = BayesRfClassifier()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Random Forest Classifier", view_link=f"/algo_rf_classifier/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_rf_classifier.view_bayesrfclassifier",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_rf_classifier(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesRfClassifier.objects.get(id=algo_id)
    except BayesRfClassifier.DoesNotExist:
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
        "import_data_target": '/algo_rf_classifier/import',
        "predict_data_target": '/algo_rf_classifier/predict',
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var,
        "train_config": train_config,
    }
    try:
        context['bayes_history'] = json.loads(algorithm_.training_history)
        context['h_para'] = json.loads(algorithm_.hyper_parameters)
        context['f_imp'] = json.loads(algorithm_.feature_importance)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_rf_classifier/main.html", context)


@permission_required("algo_rf_classifier.change_bayesrfclassifier")
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
    algorithm_ = BayesRfClassifier.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"Random Forest Classifier #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"rf_classifier_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_rf_classifier.change_bayesrfclassifier")
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
               "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_rf_classifier.change_bayesrfclassifier",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesRfClassifier.objects.get(id=algo_id)
    except BayesRfClassifier.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = column.y_column = False
        column.save()
    return redirect(f"/algo_rf_classifier/{algorithm_.id}")


def one_hot(labels):
    u, i = np.unique(labels, return_inverse=True)
    labels_1h = np.eye(u.shape[0])[i]
    class_dict_ = dict(zip(u, range(len(u))))
    return class_dict_, labels_1h


@permission_required("algo_rf_classifier.change_bayesrfclassifier")
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

    if train.cleaned_data['max_depth_min'] >= train.cleaned_data['max_depth_max']:
        context = {"color": "warning", "content": "The interval of max depth is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if train.cleaned_data['max_leaf_nodes_min'] >= train.cleaned_data['max_leaf_nodes_max']:
        context = {"color": "warning", "content": "The interval of max leaf nodes is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if train.cleaned_data['n_estimators_min'] >= train.cleaned_data['n_estimators_max']:
        context = {"color": "warning", "content": "The interval of 'number of trees' is not valid."}
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
        hyper_parameters = {
            'max_depth': (train.cleaned_data['max_depth_min'], train.cleaned_data['max_depth_max']),
            'max_leaf_nodes': (train.cleaned_data['max_leaf_nodes_min'], train.cleaned_data['max_leaf_nodes_max']),
            'n_estimators': (train.cleaned_data['n_estimators_min'], train.cleaned_data['n_estimators_max']),
        }
        fpr_poly_ = np.linspace(0, 1, 200)

        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            models_, histories, auc_s = [], [], []
            auc = {name: np.zeros(5).tolist() for name in class_dict.keys()}
            tpr_poly_ = np.zeros((y_1h.shape[1], 5, 200))
            hyper_parameters_list = []
            feature_importance_list = []

            for (train_index, valid_index), k in zip(k_fold.split(x), range(5)):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                y_1h_train, y_1h_valid = y_1h[train_index], y_1h[valid_index]

                def bayes_rf_5_fold(max_depth, max_leaf_nodes, n_estimators):
                    rf = RandomForestClassifier(
                        criterion=train.cleaned_data['criterion'],
                        max_depth=int(max_depth), max_leaf_nodes=int(max_leaf_nodes), n_estimators=int(n_estimators),
                        random_state=train.cleaned_data['random_seed']
                    )
                    rf.fit(x_train, y_train)
                    y_train_hat = rf.predict_proba(x_train)
                    auc_in_bayes = np.mean([roc_auc_score(y_1h_train[:, i], y_train_hat[:, i])
                                            for i in range(y_1h.shape[1])])
                    return auc_in_bayes

                optimizer = BayesianOptimization(f=bayes_rf_5_fold, pbounds=hyper_parameters,
                                                 random_state=train.cleaned_data['random_seed'])
                optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                                   n_iter=train.cleaned_data['bayes_iteration_times'])
                history = {i: res for i, res in enumerate(optimizer.res)}
                histories.append(history)
                mdl = RandomForestClassifier(
                    criterion=train.cleaned_data['criterion'],
                    max_depth=int(optimizer.max['params']['max_depth']),
                    max_leaf_nodes=int(optimizer.max['params']['max_leaf_nodes']),
                    n_estimators=int(optimizer.max['params']['n_estimators']),
                    random_state=train.cleaned_data['random_seed']
                )
                mdl.fit(x_train, y_train)
                models_.append(mdl)
                y_valid_hat = mdl.predict_proba(x_valid)
                for name, i in class_dict.items():
                    auc[name][k] = roc_auc_score(y_1h_valid[:, i], y_valid_hat[:, i])
                    auc[name][k] = round(float(auc[name][k]), 3)
                    fpr, tpr, _ = roc_curve(y_1h_valid[:, i], y_valid_hat[:, i])
                    tpr_poly_[i, k, :] = np.interp(fpr_poly_, fpr, tpr)
                hyper_parameters_list.append(
                    {'max_depth': mdl.max_depth, 'max_leaf_nodes': mdl.max_leaf_nodes,
                     'n_estimators': mdl.n_estimators}
                )
                feature_importance_list.append(
                    {name: weight for name, weight in zip(x_col, mdl.feature_importances_)}
                )

            f, fig = io.StringIO(), plt.figure()
            for name, i in class_dict.items():
                mean_roc = np.mean(tpr_poly_, axis=1)[i]
                mean_auc = np.mean(auc[name])
                range_auc = np.maximum(np.max(auc[name]) - mean_auc, mean_auc - np.min(auc[name]))
                plt.plot(fpr_poly_, mean_roc, label=f'{name} (AUC = {mean_auc.round(3)} ± {range_auc.round(3)})')

            intermediate_paper_handle = ContentFile(pickle.dumps(models_))
            new_paper = Paper(user=req.user, role=3, name=f'Random Forest Classifier #{algorithm_.id} Model')
            new_paper.file.save(f'rf_classifier_{algorithm_.id}_model.pkl', intermediate_paper_handle)
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
            algorithm_.feature_importance = json.dumps(feature_importance_list, ensure_ascii=False)

        elif mode == "split":
            x_train, x_valid, y_train, y_valid, y_1h_train, y_1h_valid = train_test_split(
                x, y, y_1h, train_size=0.8, shuffle=True, random_state=train.cleaned_data['random_seed'])

            def bayes_rf_split(max_depth, max_leaf_nodes, n_estimators):
                rf = RandomForestClassifier(
                    criterion=train.cleaned_data['criterion'],
                    max_depth=int(max_depth), max_leaf_nodes=int(max_leaf_nodes), n_estimators=int(n_estimators),
                    random_state=train.cleaned_data['random_seed']
                )
                rf.fit(x_train, y_train)
                y_train_hat = rf.predict_proba(x_train)
                auc_in_bayes = np.mean([roc_auc_score(y_1h_train[:, i], y_train_hat[:, i])
                                        for i in range(y_1h.shape[1])])
                return auc_in_bayes

            optimizer = BayesianOptimization(f=bayes_rf_split, pbounds=hyper_parameters,
                                             random_state=train.cleaned_data['random_seed'])
            optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                               n_iter=train.cleaned_data['bayes_iteration_times'])
            history = {i: res for i, res in enumerate(optimizer.res)}
            mdl = RandomForestClassifier(
                criterion=train.cleaned_data['criterion'],
                max_depth=int(optimizer.max['params']['max_depth']),
                max_leaf_nodes=int(optimizer.max['params']['max_leaf_nodes']),
                n_estimators=int(optimizer.max['params']['n_estimators']),
                random_state=train.cleaned_data['random_seed']
            )
            mdl.fit(x_train, y_train)
            y_valid_hat = mdl.predict_proba(x_valid)
            hyper_parameters = {'max_depth': mdl.max_depth, 'max_leaf_nodes': mdl.max_leaf_nodes,
                                'n_estimators': mdl.n_estimators}
            feature_importance_ = {name: weight for name, weight in zip(x_col, mdl.feature_importances_)}
            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)
            algorithm_.feature_importance = json.dumps(feature_importance_, ensure_ascii=False)
            auc = {}
            f, fig = io.StringIO(), plt.figure()

            for name, i in class_dict.items():
                auc[name] = roc_auc_score(y_1h_valid[:, i], y_valid_hat[:, i])
                auc[name] = round(float(auc[name]), 3)
                fpr, tpr, _ = roc_curve(y_1h_valid[:, i], y_valid_hat[:, i])
                tpr_poly_ = np.interp(fpr_poly_, fpr, tpr)
                plt.plot(fpr_poly_, tpr_poly_, label=f'{name} (AUC = {auc[name].__round__(3)})')

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Random Forest Classifier #{algorithm_.id} Model')
            new_paper.file.save(f'rf_classifier_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
            algorithm_.training_history = json.dumps(history, ensure_ascii=False)
            algorithm_.auc = json.dumps(auc, ensure_ascii=False)

            plt.plot([0, 1], [0, 1], linestyle='--', lw=1.25, color='b', label='Chance')
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend(loc=4)
            fig.savefig(f, format='svg')
            plt.close(fig)
            algorithm_.roc_curve = f.getvalue()

        else:  # mode == "full_train"
            def bayes_rf_full_train(max_depth, max_leaf_nodes, n_estimators):
                rf = RandomForestClassifier(
                    criterion=train.cleaned_data['criterion'],
                    max_depth=int(max_depth), max_leaf_nodes=int(max_leaf_nodes), n_estimators=int(n_estimators),
                    random_state=train.cleaned_data['random_seed']
                )
                rf.fit(x, y)
                y_hat = rf.predict_proba(x)
                auc_in_bayes = np.mean([roc_auc_score(y_1h[:, i], y_hat[:, i])
                                        for i in range(y_1h.shape[1])])
                return auc_in_bayes
            optimizer = BayesianOptimization(f=bayes_rf_full_train, pbounds=hyper_parameters,
                                             random_state=train.cleaned_data['random_seed'])
            optimizer.maximize(init_points=train.cleaned_data['bayes_init_try_times'],
                               n_iter=train.cleaned_data['bayes_iteration_times'])
            history = {i: res for i, res in enumerate(optimizer.res)}
            mdl = RandomForestClassifier(
                criterion=train.cleaned_data['criterion'],
                max_depth=int(optimizer.max['params']['max_depth']),
                max_leaf_nodes=int(optimizer.max['params']['max_leaf_nodes']),
                n_estimators=int(optimizer.max['params']['n_estimators']),
                random_state=train.cleaned_data['random_seed']
            )
            mdl.fit(x, y)
            hyper_parameters = {'max_depth': mdl.max_depth, 'max_leaf_nodes': mdl.max_leaf_nodes,
                                'n_estimators': mdl.n_estimators}
            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)
            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'Random Forest Classifier #{algorithm_.id} Model')
            new_paper.file.save(f'rf_classifier_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper
            algorithm_.training_history = json.dumps(history, ensure_ascii=False)
            feature_importance_ = {name: weight for name, weight in zip(x_col, mdl.feature_importances_)}
            algorithm_.feature_importance = json.dumps(feature_importance_, ensure_ascii=False)

        algorithm_.class_dict = json.dumps(class_dict, ensure_ascii=False)
        algorithm_.mode = mode
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
        raise e
        # return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_rf_classifier.change_bayesrfclassifier",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = BayesRfClassifier.objects.get(id=algo_id)
    except BayesRfClassifier.DoesNotExist:
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
    algorithm_.feature_importance = str()
    algorithm_.roc_curve = str()
    algorithm_.save()
    return redirect(f"/algo_rf_classifier/{algorithm_.id}")


def most_frequent_item(a: np.array):
    a = a.tolist()
    return max(set(a), key=a.count)


@permission_required("algo_rf_classifier.change_bayesrfclassifier",
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
    algorithm_ = BayesRfClassifier.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
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
    new_paper = Paper(user=req.user, role=4, name=f"Random Forest Classifier #{algorithm_.id} Predict")
    new_paper.file.save(f"rf_classifier_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_rf_classifier/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
