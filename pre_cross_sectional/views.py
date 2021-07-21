import pickle
from scipy.stats import mode as math_mode
import numpy as np
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
    algorithm = forms.ModelChoiceField(PreProcessing.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = PreProcessing.objects.get(id=algo_id)


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
        self.fields['paper'].queryset = Paper.objects.filter(Q(role=1) | Q(role=2), user=user)


class Profile(PublicAlgorithm):
    pass


@permission_required("pre_cross_sectional.add_preprocessing",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_csp(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = PreProcessing()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Pre-processing for Cross-sectional Data",
        view_link=f"/pre_cross_sectional/{new_algorithm.id}"
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("pre_cross_sectional.change_preprocessing",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
@csrf_exempt
@require_POST
def generate_profile(req):
    profile_sheet = Profile(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = profile_sheet.is_valid()
    csp = profile_sheet.cleaned_data['algorithm']
    v2 = csp.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    if csp.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    if not csp.dataframe:
        context = {"color": "danger", "content": "This instance doesn't contain data and variables."}
        return render(req, "task_manager/hint_widget.html", context)
    csp.step.status = 2
    csp.step.save()
    try:
        with open(csp.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        profile = ProfileReport(dataframe, title=f"Cross-sectional Data Pre-processing #{csp.id}",
                                plot={"dpi": 200, "image_format": "png"})
        csp.report = profile.to_html()
    except Exception as e:
        csp.step.status = 4
        csp.step.save()
        csp.error_message = str(e)
    csp.save()
    csp.step.status = 3
    csp.step.save()
    context = {"color": "success", "content": "Generate the profile successfully."}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_cross_sectional.view_preprocessing",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_profile(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = PreProcessing.objects.get(id=algo_id)
    except PreProcessing.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    return HttpResponse(algorithm_.report)


@permission_required("preprocessing.change_preprocessing")
@csrf_exempt
@require_POST
def change_note(req):
    note = Note(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = note.is_valid()
    algorithm_ = note.cleaned_data['algorithm']
    v2 = algorithm_.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    algorithm_.note = note.cleaned_data['experiment_note']
    algorithm_.save()
    context = {"color": "success", "content": "Note is changed successfully."}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("library.view_paper")
@csrf_exempt
@require_POST
def search_data(req):
    search_file = SearchFile(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = search_file.is_valid()
    algorithm_ = search_file.cleaned_data['algorithm']
    v2 = algorithm_.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    select_file = SelectFile()
    select_file.search_paper(req.user, search_file.cleaned_data['search_file'], search_file.cleaned_data['data_format'])
    select_file.link_to_algorithm(algorithm_.id)
    return HttpResponse(select_file.as_p())


@permission_required("pre_cross_sectional.change_preprocessing")
@csrf_exempt
@require_POST
def use_data(req):
    select_file = SelectFile(req.POST)
    select_file.ownership_paper(req.user)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = select_file.is_valid()
    algorithm_ = select_file.cleaned_data['algorithm']
    v2 = algorithm_.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
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
        new_paper = Paper(user=req.user, role=2, name=f"Cross-sectional Data Pre-processing #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"csp_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        for column in Column.objects.filter(algorithm=algorithm_):
            column.delete()
        for col in table.columns:
            new_column = Column(algorithm=algorithm_, name=col)
            new_column.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/pre_cross_sectional/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_cross_sectional/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_cross_sectional.change_preprocessing",
                     login_url="/task/retrieve?message=You don't have access change algorithms.&color=danger")
def confirm_error(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = PreProcessing.objects.get(id=algo_id)
    except PreProcessing.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.error_message = ""
    algorithm_.save()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_cross_sectional/{algorithm_.id}")


@permission_required("pre_cross_sectional.change_preprocessing",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_data(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = PreProcessing.objects.get(id=algo_id)
    except PreProcessing.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.dataframe.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_cross_sectional/{algorithm_.id}")


class PublicPreProcessing(forms.Form):
    algorithm = forms.ModelChoiceField(PreProcessing.objects.all(), widget=forms.HiddenInput())
    targeted_columns = forms.ModelMultipleChoiceField(Column.objects.all(), widget=forms.CheckboxSelectMultiple())

    def link_to_algorithm(self, algo_id: int):
        self.fields['algorithm'].initial = PreProcessing.objects.get(id=algo_id)
        self.fields['targeted_columns'].queryset = Column.objects.filter(algorithm_id=algo_id)

    def set_label_id(self, name: str):
        self.fields['targeted_columns'].widget.id_for_label(name)


class DropColumns(PublicPreProcessing):
    pass


class FillNa(PublicPreProcessing):
    method = forms.ChoiceField(
        choices=(
            ("ffill", "propagate last valid observation forward to next valid"),
            ("bfill", "use next valid observation to fill gap"),
            (None, "use constant to fill holes (e.g. 0)"),
        ),
        required=False,
        widget=forms.Select({"class": "form-select"}),
    )
    quick_constant = forms.ChoiceField(
        choices=(
            (None, "------"),
            ('average', "The average value of the sample."),
            ('average-95', "The average value after cutting samples in both sides (5% each)."),
            ('mode', "The mode of the sample."),
            ('min', "Minimum value."),
            ('max', "Maximum value."),
        ),
        required=False,
        widget=forms.Select({"class": "form-select"}),
        help_text="For more values, obtain statistical metrics in data profile."
    )
    constant = forms.FloatField(
        widget=forms.NumberInput({"class": "form-control"}),
        required=False,
    )


def pandas_drop_column(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.drop(
        columns=[x.name for x in config.cleaned_data['targeted_columns']],
        inplace=True,
    )
    config.cleaned_data['targeted_columns'].delete()
    return dataframe


def pandas_cast(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:

    return dataframe


def pandas_fill_na(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [x.name for x in config.cleaned_data['targeted_columns']]
    if config.cleaned_data['method']:
        dataframe[columns] = dataframe[columns].fillna(method=config.cleaned_data['method'], axis=0)
    else:
        if config.cleaned_data['quick_constant'] == 'average':
            na_value = dataframe[columns].mean(axis=0).to_dict()
        elif config.cleaned_data['quick_constant'] == 'average-95':
            n = dataframe.shape[0]
            na_value = {
                col: np.mean(np.sort(dataframe[col])[round(.05 * n):round(.95 * n)])
                for col in columns
            }
        elif config.cleaned_data['quick_constant'] == 'mode':
            na_value = dict(zip(columns, math_mode(dataframe[columns]).mode.squeeze()))
        elif config.cleaned_data['quick_constant'] == 'min':
            na_value = dataframe[columns].min(axis=0).to_dict()
        elif config.cleaned_data['quick_constant'] == 'max':
            na_value = dataframe[columns].max(axis=0).to_dict()
        else:
            na_value = config.cleaned_data['constant'] or 0
        dataframe[columns] = dataframe[columns].fillna(value=na_value)
    return dataframe


preprocessing_wrapper_menu = {
    "drop_column": {"form": DropColumns, "function": pandas_drop_column},
    "fill_na": {"form": FillNa, "function": pandas_fill_na},
}


@permission_required("pre_cross_sectional.view_preprocessing",
                     login_url="/task/retrieve?message=You don't have permission to view this algorithm.&color=danger")
def view_csp(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = PreProcessing.objects.get(id=algo_id)
    except PreProcessing.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    search_file = SearchFile()
    search_file.link_to_algorithm(algo_id)
    note = Note()
    note.link_to_algorithm(algo_id)
    note.load_note(algorithm_.note)
    profile_sheet = Profile()
    profile_sheet.link_to_algorithm(algo_id)
    context = {
        "pre_csp": algorithm_, "notepad": note, "search_file": search_file, "search_result_empty": SelectFile(),
        "profile": profile_sheet,
    }
    for form_name, form_config in preprocessing_wrapper_menu.items():
        preprocessing_sheet = preprocessing_wrapper_menu[form_name]['form']()
        preprocessing_sheet.link_to_algorithm(algo_id)
        preprocessing_sheet.fields['targeted_columns'].widget.__dict__['attrs']['id'] = form_name + '_tc'
        preprocessing_sheet.fields['algorithm'].widget.__dict__['attrs']['id'] = form_name + '_algo'
        context[form_name] = preprocessing_sheet
    return render(req, "pre_cross_sectional/main.html", context)


@permission_required("preprocessing.change_preprocessing")
@csrf_exempt
@require_POST
def preprocessing_wrapper(req, form_name):
    if form_name not in preprocessing_wrapper_menu.keys():
        context = {"color": "danger", "content": "Function not found."}
        return render(req, "task_manager/hint_widget.html", context)
    preprocessing_form = preprocessing_wrapper_menu[form_name]['form'](req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = preprocessing_form.is_valid()
    csp = preprocessing_form.cleaned_data['algorithm']
    v2 = csp.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    if not csp.dataframe:
        context = {"color": "danger", "content": "This instance doesn't contain data and variables."}
        return render(req, "task_manager/hint_widget.html", context)
    csp.step.status = 2
    csp.step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        with open(csp.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        dataframe = preprocessing_wrapper_menu[form_name]['function'](preprocessing_form, dataframe)
        with open(csp.dataframe.file.path, "wb") as f:
            pickle.dump(dataframe, f)
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        csp.step.status = 4
        csp.step.save()
        csp.error_message = str(e)
        csp.save()
        raise e
        # context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_cross_sectional/{csp.id}"}
        # return render(req, "task_manager/hint_widget.html", context)
    csp.step.status = 3
    csp.step.save()
    context = {"color": "success", "content": "The dataset has been updated.", "refresh": f"/pre_cross_sectional/{csp.id}"}
    return render(req, "task_manager/hint_widget.html", context)
