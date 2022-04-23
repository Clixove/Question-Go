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

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(Resampling.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = Resampling.objects.get(id=algo_id)


class VariablePicker(PublicAlgorithm):
    Dependent_Variable_Y = forms.ModelChoiceField(
        Column.objects.all(), widget=forms.Select({"class": "form-select"}), initial=""
    )

    def load_choices(self, algorithm):
        columns = Column.objects.filter(algorithm=algorithm)
        self.fields['Dependent_Variable_Y'].queryset = columns
        self.fields['Dependent_Variable_Y'].initial = columns.last()


class Train(PublicAlgorithm):
    sample_size = forms.IntegerField(
        min_value=1, widget=forms.NumberInput({'class': 'form-control'}),
        help_text='Sample size for each class.'
    )


@permission_required("pre_resampling.add_resampling",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_resampling(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = Resampling()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Re-sampling", view_link=f"/pre_resampling/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("pre_resampling.view_resampling",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_resampling(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Resampling.objects.get(id=algo_id)
    except Resampling.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    variable_picker = VariablePicker()
    variable_picker.link_to_algorithm(algorithm_.id)
    variable_picker.load_choices(algorithm_)
    y_var = algorithm_.column_set.filter(y_column=True).first()
    train_config = Train()
    train_config.link_to_algorithm(algorithm_.id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/pre_resampling/import',
        "predict_data_target": '/pre_resampling/predict',
        "variable_picker": variable_picker, "y_var": y_var,
        "train_config": train_config,
    }
    try:
        context['class_dict'] = json.loads(algorithm_.class_dict)
    except json.JSONDecodeError:
        pass
    return render(req, "pre_resampling/main.html", context)


@permission_required("pre_resampling.change_resampling")
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
    algorithm_ = Resampling.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"Re-sampling #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"resampling_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/pre_resampling/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_resampling/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_resampling.change_resampling")
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
    column = variable_picker.cleaned_data['Dependent_Variable_Y']
    column.y_column = True
    column.save()
    try:
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        class_dict = {
            name: sub_df.shape[0]
            for name, sub_df in dataframe.groupby(column.name)
        }
        algorithm_.class_dict = json.dumps(class_dict, ensure_ascii=False)
        algorithm_.save()
    except Exception as e:
        context = {'color': 'danger', 'content': f'Errors occurred when counting samples of each class. {e}'}
        return render(req, "task_manager/hint_widget.html", context)
    context = {"color": "success", "content": "Set variables successfully.",
               "refresh": f"/pre_resampling/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_resampling.change_resampling",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):  # SPECIFIED
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Resampling.objects.get(id=algo_id)
    except Resampling.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_, y_column=True):
        column.y_column = False
        column.save()
    algorithm_.class_dict = str()
    algorithm_.save()
    return redirect(f"/pre_resampling/{algorithm_.id}")


@permission_required("pre_resampling.change_resampling")
@csrf_exempt
@require_POST
def train_model(req):  # SPECIFIED
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
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        y_col = Column.objects.get(algorithm=algorithm_, y_column=True).name
        samples_index = np.empty(shape=0, dtype=np.int32)
        for name, sub_df in dataframe.groupby(y_col):
            sample_index = np.random.choice(sub_df.index.tolist(), size=train.cleaned_data['sample_size'])
            samples_index = np.hstack([samples_index, sample_index])
        dataframe = dataframe.loc[samples_index, :]

        intermediate_paper_handle = ContentFile(pickle.dumps(dataframe))
        new_paper = Paper(user=req.user, role=4, name=f'Re-sampling #{algorithm_.id} Predict')
        new_paper.file.save(f'resampling_{algorithm_.id}_predict.pkl', intermediate_paper_handle)
        new_paper.save()
        step.predicted_data = new_paper
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/pre_resampling/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/pre_resampling/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)
