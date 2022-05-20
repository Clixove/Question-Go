import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from question_go_v2.settings import TIME_ZONE
from .models import *
from .register import algorithm_registry


class RenameTask(forms.Form):
    new_name = forms.CharField(max_length=64, widget=forms.TextInput({"class": "form-control"}))


def view_main_page(req):
    return render(req, "task_manager/main.html")


@permission_required("task_manager.view_task", login_url=f"/main?message=No permission to view task.&color=danger")
def view_instances(req):
    try:
        default_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        default_task = None
    context = {
        "opened_task": default_task,
        "tasks": Task.objects.filter(user=req.user).exclude(openedtask__user=req.user),
        "timezone": TIME_ZONE,
        "rename_task_form": RenameTask(),
    }
    return render(req, "task_manager/projects.html", context)


@permission_required("task_manager.view_task",
                     login_url="/task/instances?message=You don't have permission to view this task.&color=danger")
def view_task(req, task_id):
    try:
        this_task = Task.objects.get(id=task_id, openedtask__user=req.user)
    except Task.DoesNotExist:
        return redirect("/task/instances?message=This task is closed or isn't published to you.&color=danger")
    context = {
        "task": this_task,
        "spinner_color_picker": ["primary", "warning", "success", "danger"],
        'registry': algorithm_registry,
    }
    return render(req, "task_manager/steps.html", context)


@permission_required("task_manager.add_openedtask",
                     login_url="/task/instances?message=You don't have permission to open tasks.&color=danger")
def open_task(req, task_id):
    try:
        open_task_ = OpenedTask.objects.get(user=req.user)
        open_task_.task_id = task_id
        open_task_.save()
    except OpenedTask.DoesNotExist:
        open_task_ = OpenedTask(user=req.user, task_id=task_id)
        open_task_.save()
    return redirect(f"/task/{task_id}?message=Opened successfully.&color=success")


@permission_required("task_manager.delete_openedtask",
                     login_url="/task/instances?message=You don't have permission to close tasks.&color=danger")
def close_task(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user)
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=All of your tasks have been closed.&color=warning")
    if opened_task.task.busy():
        return redirect("/task/instances?message=This task is busy, so it cannot be closed.&color=danger")
    opened_task.delete()
    return redirect("/task/instances?message=Close successfully.&color=success")


@permission_required("task_manager.delete_task",
                     login_url="/task/instances?message=You don't have permission to delete projects.&color=danger")
def delete_task(req, task_id):
    try:
        task = Task.objects.get(id=task_id, user=req.user)
    except Task.DoesNotExist:
        return redirect("/task/instances?message=This task does not exist.&color=warning")
    if task.busy():
        return redirect("/task/instances?message=This task is busy so cannot be deleted.&color=danger")
    task.delete()
    return redirect("/task/instances?message=Delete successfully.&color=success")


class AddTask(forms.Form):
    name = forms.CharField(
        widget=forms.TextInput({"class": "form-control"}),
        max_length=64,
    )
    open_after_created = forms.BooleanField(
        widget=forms.Select({"class": "form form-select"}, choices=[(True, "Yes"), (False, "No")]),
        required=False, initial=True,
    )


@permission_required("task_manager.view_task",
                     login_url="/task/instances?message=You don't have permission to view projects.&color=danger")
def view_add_task(req):
    context = {"add_task_form": AddTask()}
    return render(req, "task_manager/new_project.html", context)


@permission_required("task_manager.add_task",
                     login_url="/task/instances?message=You don't have permission to add projects.&color=danger")
@require_POST
@csrf_exempt
def add_task(req):
    at = AddTask(req.POST)
    if not at.is_valid():
        return redirect("/task/new?message=Submission is not valid.&color=danger")
    new_task = Task(user=req.user, name=at.cleaned_data['name'])
    new_task.save()
    if at.cleaned_data['open_after_created']:
        try:
            new_opened_task = OpenedTask.objects.get(user=req.user)
            new_opened_task.task = new_task
        except OpenedTask.DoesNotExist:
            new_opened_task = OpenedTask(user=req.user, task=new_task)
        new_opened_task.save()
        return redirect(f"/task/{new_task.id}?message=Task added successfully.&color=success")
    return redirect("/task/instances?message=Task added successfully.&color=success")


@permission_required("task_manager.view_task", login_url="/main?message=No permission to view tasks.&color=danger")
def retrieve_task(req):
    try:
        opened_task_id = OpenedTask.objects.get(user=req.user).task_id
        return view_task(req, task_id=opened_task_id)
    except OpenedTask.DoesNotExist:
        if Task.objects.filter(user=req.user).exists():
            return view_instances(req)
        else:
            return view_add_task(req)


@permission_required("task_manager.change_task",
                     login_url="/task/instances?message=You don't have permission to change projects.&color=danger")
@require_POST
@csrf_exempt
def rename_task(req, task_id):
    try:
        task = Task.objects.get(id=task_id, user=req.user)
    except Task.DoesNotExist:
        return redirect("/task/instances?message=This task does not exist.&color=warning")
    rt = RenameTask(req.POST)
    if not rt.is_valid():
        return redirect("/task/instances?message=Submission is not valid.&color=danger")
    task.name = rt.cleaned_data['new_name']
    task.save()
    return redirect("/task/instances?message=Rename successfully.&color=success")


@permission_required("task_manager.delete_step",
                     login_url="/task/instances?message=You have no access to delete this step.&color=danger")
def delete_step(req, step_id):
    try:
        opened_task_id = OpenedTask.objects.get(user=req.user).task_id
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=Entrance page not permitted.&color=danger")
    try:
        step = Step.objects.get(id=step_id, task_id=opened_task_id)
    except Step.DoesNotExist:
        return redirect(f"/task/{opened_task_id}?message=This step doesn't exist.&color=danger")
    if step.status == 2:
        return redirect(f"/task/{opened_task_id}?message=This step is running so cannot be deleted.&color=danger")
    if step.linked_data: step.linked_data.delete()
    if step.predicted_data: step.predicted_data.delete()
    step.delete()
    return redirect(f"/task/{opened_task_id}?message=Delete successfully.&color=success")


class Note(forms.Form):
    step = forms.ModelChoiceField(Step.objects.all(), widget=forms.HiddenInput())
    note = forms.CharField(widget=forms.Textarea({"class": "form-control"}), required=False, label="")


def display_note(step): return Note(initial={'step': step, 'note': step.note})


@csrf_exempt
@require_POST
@permission_required("task_manager.change_step")
def change_note(req):
    note = Note(req.POST)
    if not note.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    step = note.cleaned_data['step']
    if not step.task.user == req.user:
        context = {"color": "danger", "content": "Do not have permission to change this step."}
        return render(req, "task_manager/hint_widget.html", context)
    step.note = note.cleaned_data['note']
    step.save()
    context = {"color": "success", "content": "Note is changed successfully."}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("task_manager.change_step",
                     login_url="/task/retrieve?message=You don't have access change this step.&color=danger")
def confirm_error(req, step_id):
    try:
        step = Step.objects.get(id=step_id, task__user=req.user)
    except Step.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    step.error_message = str()
    step.status = 1
    step.save()
    return redirect(step.view_link)


class DataPicker(forms.Form):
    step = forms.ModelChoiceField(Step.objects.all(), widget=forms.HiddenInput())
    paper = forms.ModelChoiceField(Paper.objects.all(), widget=forms.Select({'class': 'form-select'}), empty_label=None)
    data_format = forms.ChoiceField(choices=[(1, "Spreadsheet [*.xlsx]"), (2, "Binary [*.pkl]")],
                                    widget=forms.Select({"class": "form-select"}))

    def load_choices(self, user, search):
        queryset = Paper.objects.filter(user=user, name__contains=search)
        self.fields['step'].queryset = Step.objects.filter(task__user=user)
        self.fields['paper'].queryset = queryset


class DataSearch(forms.Form):
    step = forms.ModelChoiceField(Step.objects.all(), widget=forms.HiddenInput())
    filename = forms.CharField(
        widget=forms.TextInput({
            'class': 'form-control',
            'placeholder': 'Input and press `Enter` to search file.'
        }),
        required=False
    )


def display_data_picker(step):
    return DataSearch(initial={'step': step})


@permission_required("library.view_paper")
@csrf_exempt
@require_POST
def search_data(req):
    ds = DataSearch(req.POST)
    if not ds.is_valid():
        return HttpResponse(DataPicker(req.user, str()))
    step = ds.cleaned_data['step']
    data_picker = DataPicker()
    data_picker.load_choices(req.user, ds.cleaned_data['filename'])
    data_picker.fields['step'].initial = step
    return HttpResponse(data_picker.as_p())


@permission_required("task_manager.change_step",
                     login_url="/task/retrieve?message=You don't have access change this step.&color=danger")
@csrf_exempt
@require_POST
def import_training_set_v2(req):
    data_picker = DataPicker(req.POST)
    data_picker.load_choices(req.user, str())
    if not data_picker.is_valid():
        return None, None, data_picker.errors
    paper = data_picker.cleaned_data['paper']
    step = data_picker.cleaned_data['step']
    step.linked_data = paper
    step.status = 2
    step.save()
    try:
        if data_picker.cleaned_data['data_format'] == '1':
            table = pd.read_excel(paper.file.path, sheet_name=0)
            table.columns = [x.__str__() for x in table.columns]
        else:
            table = pd.read_pickle(paper.file.path)
    except Exception as e:
        step.status = 4
        step.save()
        return None, step, e.__str__()
    return table, step, None


@permission_required("task_manager.change_step",
                     login_url="/task/retrieve?message=You don't have access change this step.&color=danger")
@csrf_exempt
@require_POST
def import_predicting_set_v2(req):
    data_picker = DataPicker(req.POST)
    data_picker.load_choices(req.user, str())
    if not data_picker.is_valid():
        return None, None, data_picker.errors
    paper = data_picker.cleaned_data['paper']
    step = data_picker.cleaned_data['step']
    try:
        if data_picker.cleaned_data['data_format'] == '1':
            table = pd.read_excel(paper.file.path, sheet_name=0)
            table.columns = [x.__str__() for x in table.columns]
        else:
            table = pd.read_pickle(paper.file.path)
    except Exception as e:
        return None, step, e.__str__()
    return table, step, None


@permission_required("task_manager.change_step",
                     login_url="/task/retrieve?message=You don't have access change this step.&color=danger")
def delete_data(req, step_id):
    try:
        step = Step.objects.get(id=step_id, task__user=req.user)
    except Step.DoesNotExist:
        return redirect('/task/retrieve?message=You don\'t have access change this step.&color=danger')
    step.status = 1
    step.linked_data = None
    step.save()
    return redirect(step.view_link)


@permission_required("task_manager.change_step",
                     login_url="/task/retrieve?message=You don't have access change this step.&color=danger")
def delete_predicted(req, step_id):
    try:
        step = Step.objects.get(id=step_id, task__user=req.user)
    except Step.DoesNotExist:
        return redirect('/task/retrieve?message=You don\'t have access change this step.&color=danger')
    step.status = 1
    step.predicted_data = None
    step.save()
    return redirect(step.view_link)
