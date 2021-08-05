import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(Normalization.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = Normalization.objects.get(id=algo_id)


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

    def search_paper(self, user, search_query):
        paper_queryset = Paper.objects.filter(name__contains=search_query, user=user)
        self.fields['paper'].queryset = paper_queryset
        self.fields['paper'].initial = paper_queryset.first()

    def ownership_paper(self, user):
        self.fields['paper'].queryset = Paper.objects.filter(user=user)


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
    select_file.search_paper(req.user, search_file.cleaned_data['search_file'])
    select_file.link_to_algorithm(algorithm_.id)
    return HttpResponse(select_file.as_p())


@permission_required("pre_norm.change_normalization")
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
        if select_file.cleaned_data['data_format'] == '1':
            table = pd.read_excel(paper.file.path)
        else:
            table = pd.read_pickle(paper.file.path)
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2, name=f"Normalization #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"norm_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/pre_norm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have access change algorithms.&color=danger")
def confirm_error(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.error_message = ""
    algorithm_.save()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_norm/{algorithm_.id}")


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_data(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.dataframe.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_norm/{algorithm_.id}")


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
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    search_file = SearchFile()
    search_file.link_to_algorithm(algo_id)
    note = Note()
    note.link_to_algorithm(algo_id)
    note.load_note(algorithm_.note)
    norm_sheet = NormCore()
    norm_sheet.link_to_algorithm(algo_id)
    context = {
        "pre_csp": algorithm_, "notepad": note, "search_file": search_file, "search_result_empty": SelectFile(),
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

    if config.cleaned_data['method'] == 'S':
        op = StandardScaler()
    else:  # config.cleaned_data['method'] == 'M'
        op = MinMaxScaler()
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
        new_predict.file.save(f"norm_{algorithm_.id}_predict.pkl", intermediate_paper_handle)
        new_predict.save()


        intermediate_paper_handle = ContentFile(pickle.dumps(op))
        new_model = Paper(user=req.user, role=3, name=f"Normalization #{algorithm_.id} Model")
        new_model.file.save(f"norm_{algorithm_.id}_model.pkl", intermediate_paper_handle)
        new_model.save()
        algorithm_.model = new_model
        algorithm_.predict = new_predict
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_norm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_norm.view_normalization",
                     login_url="/task/retrieve?message=You don't have permission to view this algorithm.&color=danger")
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
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        paper = select_file.cleaned_data['paper']
        if select_file.cleaned_data['data_format'] == '1':
            table = pd.read_excel(paper.file.path)
        else:
            with open(paper.file.path, "rb") as f:
                table = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x=True)]
        table[x_col] = model.transform(table[x_col])
    except Exception as e:
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"Normalization #{algorithm_.id} Transformed")
    intermediate_pickle_handler = ContentFile(pickle.dumps(table))
    new_paper.file.save(f"norm_{algorithm_.id}_predict.pkl", intermediate_pickle_handler)
    new_paper.save()
    algorithm_.predict = new_paper
    algorithm_.save()
    context = {"color": "success", "content": "Prediction completed.", "refresh": f"/pre_norm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model.delete()
    return redirect(f"/pre_norm/{algorithm_.id}")


@permission_required("pre_norm.change_normalization",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def clear_predict(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = Normalization.objects.get(id=algo_id)
    except Normalization.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.predict.delete()
    return redirect(f"/pre_norm/{algorithm_.id}")
