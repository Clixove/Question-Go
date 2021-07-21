import pickle

import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pandas_profiling import ProfileReport

from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(Description.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = Description.objects.get(id=algo_id)


class Note(PublicAlgorithm):
    experiment_note = forms.CharField(widget=forms.Textarea({"class": "form-control"}), required=False, label="")

    def load_note(self, content):
        self.fields['experiment_note'].initial = content


class SearchFile(PublicAlgorithm):
    search_file = forms.CharField(max_length=64, widget=forms.TextInput({"class": "form-control"}), label="")


class SelectFile(PublicAlgorithm):
    paper = forms.ModelChoiceField(Paper.objects.none(), widget=forms.Select({"class": "form-select"}),
                                   label="Searching Result", empty_label=None,
                                   help_text="Choose one from searching results.")
    data_format = forms.ChoiceField(choices=[(1, "Spreadsheet [*.xlsx]"), (2, "Binary [*.pkl]")],
                                    widget=forms.Select({"class": "form-select"}))

    def search_paper(self, user, search_query, role):
        paper_queryset = Paper.objects.filter(name__contains=search_query, user=user, role=role)
        self.fields['paper'].queryset = paper_queryset
        self.fields['paper'].initial = paper_queryset.first()

    def ownership_paper(self, user):
        self.fields['paper'].queryset = Paper.objects.filter(user=user)


class VariablePicker(PublicAlgorithm):
    Independent_Variables_X = forms.ModelMultipleChoiceField(
        Column.objects.all(), widget=forms.CheckboxSelectMultiple(), label="Variables for analysis"
    )

    def load_choices(self, algorithm):
        self.fields['Independent_Variables_X'].queryset = Column.objects.filter(algorithm=algorithm)


class Profile(PublicAlgorithm):
    pass


@permission_required("pre_descriptive.add_description",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_dp(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = Description()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Descriptive Statistics", view_link=f"/pre_descriptive/{new_algorithm.id}")
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("pre_descriptive.view_description",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def view_dp(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Description.objects.get(id=algo_id)
    except Description.DoesNotExist:
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
    variable_picker.link_to_algorithm(algo_id)
    variable_picker.load_choices(algorithm_)
    profile_sheet = Profile()
    profile_sheet.link_to_algorithm(algo_id)
    context = {
        "pre_dp": algorithm_, "notepad": note, "search_file": search_file, "search_result_empty": SelectFile(),
        "x_var": Column.objects.filter(algorithm=algorithm_, x_column=True), "variable_picker": variable_picker,
        "profile": profile_sheet,
    }
    return render(req, "pre_descriptive/main.html", context)


@permission_required("pre_descriptive.change_description")
@csrf_exempt
@require_POST
def change_note(req):
    note = Note(req.POST)
    if not note.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algo_dp = note.cleaned_data['algorithm']
    if not algo_dp.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    algo_dp.note = note.cleaned_data['experiment_note']
    algo_dp.save()
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


@permission_required("pre_descriptive.change_description")
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
        if select_file.cleaned_data['data_format'] == 1:
            table = pd.read_excel(paper.file.path)
        else:
            with open(paper.file.path, "rb") as f:
                table = pickle.load(f)
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2, name=f"Descriptive Statistics #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"dp_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/pre_descriptive/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_descriptive/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_descriptive.change_description",
                     login_url="/task/retrieve?message=You don't have access change algorithms.&color=danger")
def confirm_error(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Description.objects.get(id=algo_id)
    except Description.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.error_message = ""
    algorithm_.save()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_descriptive/{algorithm_.id}")


@permission_required("pre_descriptive.change_description",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_data(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Description.objects.get(id=algo_id)
    except Description.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.dataframe.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_descriptive/{algorithm_.id}")


@permission_required("pre_descriptive.change_description")
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
    context = {"color": "success", "content": "Set variables successfully.",
               "refresh": f"/pre_descriptive/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_descriptive.change_description",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Description.objects.get(id=algo_id)
    except Description.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = False
        column.save()
    return redirect(f"/pre_descriptive/{algorithm_.id}")


@permission_required("pre_descriptive.change_description",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
@csrf_exempt
@require_POST
def generate_profile(req):
    profile_sheet = Profile(req.POST)
    if not profile_sheet.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator START ----------
    algorithm_ = profile_sheet.cleaned_data['algorithm']
    if not algorithm_.open_permission(req.user):
        context = {"color": "danger", "content": "You don't have access to this algorithm."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator END   ----------
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    if not (algorithm_.dataframe and Column.objects.filter(algorithm=algorithm_, x_column=True).exists()):
        context = {"color": "danger", "content": "This instance doesn't contain data and variables."}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.step.status = 2
    algorithm_.step.save()
    try:
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        variables = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        profile = ProfileReport(dataframe[variables], title=f"Descriptive Statistics #{algorithm_.id}",
                                plot={"dpi": 200, "image_format": "png"})
        algorithm_.report = profile.to_html()
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "Generate the profile successfully."}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_descriptive.view_description",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_profile(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Description.objects.get(id=algo_id)
    except Description.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    return HttpResponse(algorithm_.report)
