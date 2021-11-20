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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(MyDBSCAN.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = MyDBSCAN.objects.get(id=algo_id)


class VariablePicker(PublicAlgorithm):
    Independent_Variables_X = forms.ModelMultipleChoiceField(
        Column.objects.all(), widget=forms.CheckboxSelectMultiple()
    )

    def load_choices(self, algorithm):
        columns = Column.objects.filter(algorithm=algorithm)
        self.fields['Independent_Variables_X'].queryset = columns


class Train(PublicAlgorithm):
    epsilon = forms.FloatField(
        min_value=0, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='The "ε" argument for DBSCAN algorithm.'
    )


@permission_required("algo_dbscan.add_mydbscan",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_dbscan(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = MyDBSCAN()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="DBSCAN", view_link=f"/algo_dbscan/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_dbscan.view_mydbscan",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_dbscan(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyDBSCAN.objects.get(id=algo_id)
    except MyDBSCAN.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    variable_picker = VariablePicker()
    variable_picker.link_to_algorithm(algorithm_.id)
    variable_picker.load_choices(algorithm_)
    x_var = algorithm_.column_set.filter(x_column=True)
    train_config = Train()
    train_config.link_to_algorithm(algorithm_.id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/algo_dbscan/import',
        "predict_data_target": '/algo_dbscan/predict',
        "variable_picker": variable_picker, "x_var": x_var,
        "train_config": train_config,
    }
    try:
        context['class_dict'] = json.loads(algorithm_.class_dict)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_dbscan/main.html", context)


@permission_required("algo_dbscan.change_mydbscan")
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
    algorithm_ = MyDBSCAN.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"DBSCAN #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"dbscan_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/algo_dbscan/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_dbscan/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_dbscan.change_mydbscan")
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
    # K Nearest Neighbour
    columns = [x.name for x in variable_picker.cleaned_data['Independent_Variables_X']]
    try:
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        neighbor = NearestNeighbors(n_neighbors=2 * len(columns))
        neighbor.fit(dataframe[columns])
        distance, _ = neighbor.kneighbors(dataframe[columns])
        nearest_distance = np.sort(distance, axis=0)[:, -1]
        f, fig = io.BytesIO(), plt.figure()
        plt.plot(nearest_distance)
        plt.ylabel("Minkowski p=2 Distance of Nearest Neighbor")
        plt.xlabel("Sample Ranking")
        fig.savefig(f, format='svg')
        plt.close(fig)
        algorithm_.knn_figure = f.getvalue().decode(encoding='utf-8')
        algorithm_.save()
    except Exception as e:
        context = {'color': 'danger', 'content': f'Errors occurred when calculating K neighbor distances. {e}'}
        return render(req, "task_manager/hint_widget.html", context)
    context = {"color": "success", "content": "Set variables successfully.",
               "refresh": f"/algo_dbscan/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_dbscan.change_mydbscan",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyDBSCAN.objects.get(id=algo_id)
    except MyDBSCAN.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = False
        column.save()
    algorithm_.knn_figure = str()
    algorithm_.save()
    return redirect(f"/algo_dbscan/{algorithm_.id}")


@permission_required("algo_dbscan.change_mydbscan")
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
    if train.cleaned_data['epsilon'] == 0:
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        x = dataframe[x_col].values
        dbscan = DBSCAN(eps=train.cleaned_data['epsilon'], min_samples=2 * len(x_col))
        dataframe['dbscan_class_labels'] = class_labels = dbscan.fit_predict(x)

        intermediate_paper_handle = ContentFile(pickle.dumps(dbscan))
        new_paper = Paper(user=req.user, role=3, name=f'DBSCAN #{algorithm_.id} Model')
        new_paper.file.save(f'dbscan_{algorithm_.id}_model.pkl', intermediate_paper_handle)
        new_paper.save()
        algorithm_.model = new_paper

        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            dataframe.to_excel(f, index=False)
        new_paper = Paper(user=req.user, role=3, name=f'DBSCAN #{algorithm_.id} Predict')
        new_paper.file.save(f'dbscan_{algorithm_.id}_predict.xlsx', table_bin)
        new_paper.save()
        step.predicted_data = new_paper

        class_names, class_counts = np.unique(class_labels, return_counts=True)
        algorithm_.class_dict = json.dumps(dict(zip(class_names.tolist(), class_counts.tolist())), ensure_ascii=False)
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_dbscan/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_dbscan/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_dbscan.change_mydbscan",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyDBSCAN.objects.get(id=algo_id)
    except MyDBSCAN.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model = None
    algorithm_.class_dict = str()
    algorithm_.save()
    return redirect(f"/algo_dbscan/{algorithm_.id}")
