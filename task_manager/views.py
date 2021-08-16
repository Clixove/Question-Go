from django import forms
from django.contrib.auth.decorators import permission_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from question_go_v2.settings import TIME_ZONE
from .models import *


class RenameTask(forms.Form):
    new_name = forms.CharField(max_length=64, widget=forms.TextInput({"class": "form-control"}))


def view_main_page(req):
    return render(req, "task_manager/main.html")


@permission_required("task_manager.view_task",
                     login_url=f"/main?message=Do not have permission to view projects.&color=danger")
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


@permission_required("task_manager.view_task", login_url="/main?message=Login required.&color=danger")
def retrieve_task(req):
    try:
        opened_task_id = OpenedTask.objects.get(user=req.user).task_id
        return redirect(f"/task/{opened_task_id}")
    except OpenedTask.DoesNotExist:
        if Task.objects.filter(user=req.user).exists():
            return redirect("/task/instances")
        else:
            return redirect("/task/new")


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
    step.delete()
    return redirect(f"/task/{opened_task_id}?message=Delete successfully.&color=success")
