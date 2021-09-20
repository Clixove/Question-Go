import io
import pickle

import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(Normalization.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = Normalization.objects.get(id=algo_id)


@permission_required("pre_norm.add_normalization",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_norm(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = Normalization()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Normalization and Zipping",
        view_link=f"/pre_norm/{new_algorithm.id}", model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("pre_norm.change_normalization")
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
    algorithm_ = Normalization.objects.get(step=step)
    try:
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2, name=f"Normalization #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"norm_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        algorithm_.save()
        for column in Column.objects.filter(algorithm=algorithm_):
            column.delete()
        for col in table.columns:
            new_column = Column(algorithm=algorithm_, name=col)
            new_column.save()
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_norm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Asynchronous Algorithm END   ----------
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


class NormCore(PublicAlgorithm):
    columns = forms.ModelMultipleChoiceField(Column.objects.all(), widget=forms.CheckboxSelectMultiple())
    method = forms.ChoiceField(
        choices=[('S', 'Standard normalization'), ('M', 'Zipping to 0~1')],
        widget=forms.Select({"class": "form-select"}),
    )


@permission_required("pre_norm.view_normalization",
                     login_url="/task/retrieve?message=You don't have permission to view this algorithm.&color=danger")
def view_norm(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    norm_sheet = NormCore()
    norm_sheet.link_to_algorithm(algo_id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/pre_norm/import',
        "norm_sheet": norm_sheet,
    }
    return render(req, "pre_norm/main.html", context)


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
@csrf_exempt
@require_POST
def train(req):
    config = NormCore(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = config.is_valid()
    algorithm_ = config.cleaned_data['algorithm']
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
    op = StandardScaler() if config.cleaned_data['method'] == 'S' else MinMaxScaler()
    columns = []
    for c_model in config.cleaned_data['columns']:
        c_model.x = True
        columns.append(c_model.name)
        c_model.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        dataframe = pd.read_pickle(algorithm_.dataframe.file.path)
        dataframe[columns] = op.fit_transform(dataframe[columns])

        intermediate_paper_handle = ContentFile(pickle.dumps(dataframe))
        new_predict = Paper(user=req.user, role=4, name=f"Normalization #{algorithm_.id} Transformed")
        new_predict.file.save(f"norm_{algorithm_.id}_transformed.pkl", intermediate_paper_handle)
        new_predict.save()

        intermediate_paper_handle = ContentFile(pickle.dumps(op))
        new_model = Paper(user=req.user, role=3, name=f"Normalization #{algorithm_.id} Model")
        new_model.file.save(f"norm_{algorithm_.id}_model.pkl", intermediate_paper_handle)
        new_model.save()
        algorithm_.transformed = new_predict
        algorithm_.model = new_model
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_norm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model = None
    algorithm_.transformed = None
    algorithm_.save()
    return redirect(f"/pre_norm/{algorithm_.id}")


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
@csrf_exempt
@require_POST
def transform(req):
    # ---------- Import Data Tool START ----------
    flag, content = task_manager.views.use_data(req, train=False)
    if flag:
        context = {'color': 'danger', 'content': 'Submission is not valid.'}
        return render(req, 'task_manager/hint_widget.html', context)
    step, table = content
    # ---------- Import Data Tool End   ----------
    algorithm_ = Normalization.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x=True)]
        table[x_col] = model.transform(table[x_col])
        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            table.to_excel(f, index=False)
    except Exception as e:
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"Normalization #{algorithm_.id} Reusing Transformed")
    new_paper.file.save(f"norm_{algorithm_.id}_predict.pkl", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.", "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
@csrf_exempt
@require_POST
def inverse_transform(req):
    # ---------- Import Data Tool START ----------
    flag, content = task_manager.views.use_data(req, train=False)
    if flag:
        context = {'color': 'danger', 'content': 'Submission is not valid.'}
        return render(req, 'task_manager/hint_widget.html', context)
    step, table = content
    # ---------- Import Data Tool End   ----------
    algorithm_ = Normalization.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    try:
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x=True)]
        table[x_col] = model.inverse_transform(table[x_col])
        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            table.to_excel(f, index=False)
    except Exception as e:
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"Normalization #{algorithm_.id} Inverse Transformed")
    new_paper.file.save(f"norm_{algorithm_.id}_predict.pkl", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.", "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
