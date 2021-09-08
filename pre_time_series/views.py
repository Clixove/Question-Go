import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.validators import MinValueValidator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import typing

from task_manager.models import OpenedTask
from .models import *


def list_union(a: typing.Iterable, b: typing.Iterable) -> list:
    a_set = set(a)
    c_set = a_set.intersection(b)
    return list(c_set)


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(TimeSeries.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = TimeSeries.objects.get(id=algo_id)


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

    def search_paper(self, user, search_query):
        paper_queryset = Paper.objects.filter(name__contains=search_query, user=user)
        self.fields['paper'].queryset = paper_queryset
        self.fields['paper'].initial = paper_queryset.first()

    def ownership_paper(self, user):
        self.fields['paper'].queryset = Paper.objects.filter(user=user)


@permission_required("pre_time_series.add_timeseries",
                     login_url="/task/retrieve?message=You don't have access to this algorithm.&color=danger")
def add_ts(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = TimeSeries()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="Pre-processing Time Series",
        view_link=f"/pre_ts/{new_algorithm.id}", model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("pre_time_series.change_timeseries")
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


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have access change algorithms.&color=danger")
def confirm_error(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    algorithm_.error_message = ""
    algorithm_.save()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


# STEP 1: Import data
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


@permission_required("pre_time_series.change_timeseries")
@csrf_exempt
@require_POST
def import_data(req):
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
    algorithm_.cached_dataframe = select_file.cleaned_data['paper']
    algorithm_.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_data(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.cached_dataframe = None
    algorithm_.from_datetime = None
    algorithm_.to_datetime = None
    algorithm_.periods = None
    algorithm_.save()
    for column in Column.objects.filter(algorithm=algorithm_):
        column.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_sheet(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.time_series_sheet = str()
    algorithm_.label_sheet = str()
    algorithm_.from_datetime = None
    algorithm_.to_datetime = None
    algorithm_.periods = None
    algorithm_.save()
    algorithm_.dataframe.delete()
    for column in Column.objects.filter(algorithm=algorithm_):
        column.delete()
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


class SelectSheet(PublicAlgorithm):
    time_series_sheet = forms.CharField(
        widget=forms.Select({"class": "form-select"}),
        help_text="The sheet containing multi-dimensional time series, usually expanded as a 2D table. "
                  "Sample index and time points are two of columns.",
    )
    labels_sheet = forms.CharField(
        widget=forms.Select({"class": "form-select"}), required=False,
        help_text="The classification or regression label corresponding to time series.")

    def load_choices(self, sheet_names):
        self.fields['time_series_sheet'].widget.choices = [(x, x) for x in sheet_names]
        self.fields['labels_sheet'].widget.choices = [(None, '')] + [(x, x) for x in sheet_names]


class SelectColumns(PublicAlgorithm):
    from_datetime = forms.DateTimeField(widget=forms.DateTimeInput({
        'class': 'form-control', 'format-value': 'YYYY-MM-DD HH:mm:ss'}))
    to_datetime = forms.DateTimeField(widget=forms.DateTimeInput({
        'class': 'form-control', 'format-value': 'YYYY-MM-DD HH:mm:ss'}))
    periods = forms.IntegerField(widget=forms.NumberInput({'class': 'form-control'}), validators=[MinValueValidator(1)])
    date = forms.ModelChoiceField(Column.objects.all())
    company_trans = forms.ModelChoiceField(Column.objects.all())
    use = forms.ModelMultipleChoiceField(Column.objects.all())
    log = forms.ModelMultipleChoiceField(Column.objects.all(), required=False)
    diff = forms.ModelMultipleChoiceField(Column.objects.all(), required=False)
    fill_na_avg = forms.ModelMultipleChoiceField(Column.objects.all(), required=False)
    company_score = forms.ModelChoiceField(Column.objects.all(), required=False)
    score = forms.ModelChoiceField(Column.objects.all(), required=False)

    def load_customized_part(self, algorithm):
        if algorithm.label_sheet:
            self.fields['company_score'].required = True
            self.fields['score'].required = True
        self.fields['date'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['company_trans'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['use'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['log'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['diff'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['fill_na_avg'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=True)
        self.fields['company_score'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=False)
        self.fields['score'].queryset = Column.objects.filter(algorithm=algorithm, belong_time_series=False)


@permission_required("pre_time_series.view_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to view this algorithm.&color=danger")
def view_ts(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    search_file = SearchFile()
    search_file.link_to_algorithm(algo_id)
    note = Note()
    note.link_to_algorithm(algo_id)
    note.load_note(algorithm_.note)
    if algorithm_.cached_dataframe:
        try:
            xl = pd.ExcelFile(algorithm_.cached_dataframe.file.path)
            select_sheet_ = SelectSheet(initial={
                'time_series_sheet': algorithm_.time_series_sheet,
                'labels_sheet': algorithm_.label_sheet,
            })
            select_sheet_.load_choices(xl.sheet_names)
        except Exception as e:
            algorithm_.cached_dataframe = None
            algorithm_.save()
            return redirect(f"/pre_ts/{algorithm_.id}?message={e}&color=warning")
    else:
        select_sheet_ = SelectSheet()
    select_sheet_.link_to_algorithm(algo_id)
    ac = SelectColumns(initial={
        'from_datetime': algorithm_.from_datetime, 'to_datetime': algorithm_.to_datetime,
        'periods': algorithm_.periods,
    })
    ac.link_to_algorithm(algo_id)
    context = {
        "pre_csp": algorithm_, "notepad": note, "search_file": search_file, "search_result_empty": SelectFile(),
        'select_sheet': select_sheet_, 'assign_column': ac,
        'time_series_columns': Column.objects.filter(algorithm=algorithm_, belong_time_series=True),
        'label_columns': Column.objects.filter(algorithm=algorithm_, belong_time_series=False)
        if algorithm_.label_sheet else None,
    }
    return render(req, "pre_time_series/main.html", context)


@csrf_exempt
@require_POST
@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def select_sheet(req):
    ss = SelectSheet(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = ss.is_valid()
    algorithm_ = ss.cleaned_data['algorithm']
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
    try:
        # ---------- Asynchronous Algorithm START ----------
        algorithm_.time_series_sheet = ss.cleaned_data['time_series_sheet']
        algorithm_.label_sheet = ss.cleaned_data['labels_sheet']
        ts_sheet = pd.read_excel(algorithm_.cached_dataframe.file.path, sheet_name=ss.cleaned_data['time_series_sheet'])
        if ss.cleaned_data['labels_sheet']:
            label_sheet = pd.read_excel(algorithm_.cached_dataframe.file.path, sheet_name=ss.cleaned_data['labels_sheet'])
        else:
            label_sheet = pd.DataFrame()
        intermediate_paper_handle = ContentFile(pickle.dumps({'time_series': ts_sheet, 'labels': label_sheet}))
        new_paper = Paper(user=req.user, role=2, name=f"Pre-processing Time Series #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"ts_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        [c.delete() for c in Column.objects.filter(algorithm=algorithm_)]
        for column in ts_sheet.columns:
            new_column = Column(algorithm=algorithm_, name=column)
            new_column.save()
        if ss.cleaned_data['labels_sheet']:
            for column in label_sheet.columns:
                new_column = Column(algorithm=algorithm_, name=column, belong_time_series=False)
                new_column.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_ts/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The sheets are successfully assigned.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def clear_transform(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")

    algorithm_.normalizers.delete()
    algorithm_.matrix.delete()

    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


@csrf_exempt
@require_POST
@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def transform(req):
    sc = SelectColumns(req.POST)

    v1 = sc.is_valid()
    algorithm_ = sc.cleaned_data['algorithm']
    v2 = algorithm_.open_permission(req.user)
    sc.load_customized_part(algorithm_)
    v3 = sc.is_valid()
    if not (v1 and v2 and v3):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if algorithm_.step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.step.status = 2
    algorithm_.step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        for column in Column.objects.filter(algorithm=algorithm_):
            column.is_date = column == sc.cleaned_data['date']
            column.is_index = (column == sc.cleaned_data['company_trans']) or (column == sc.cleaned_data['company_score'])
            column.is_label = column == sc.cleaned_data['score']
            column.use = column in sc.cleaned_data['use']
            column.log = column in sc.cleaned_data['log']
            column.diff = column in sc.cleaned_data['diff']
            column.fill_na_avg = column in sc.cleaned_data['fill_na_avg']
            column.save()
        algorithm_.from_datetime = sc.cleaned_data['from_datetime']
        algorithm_.to_datetime = sc.cleaned_data['to_datetime']
        algorithm_.periods = sc.cleaned_data['periods']
        algorithm_.save()

        with open(algorithm_.dataframe.file.path, 'rb') as f:
            intermediate_data_handle = pickle.load(f)
        x = intermediate_data_handle['time_series']
        columns = Column.objects.filter(algorithm=algorithm_)

        log_features = [z.name for z in columns.filter(log=True)]
        feat_name = [z.name for z in columns.filter(use=True)]
        x[log_features] = x[log_features].apply(lambda z: np.log(z + 1))
        mm_x = MinMaxScaler()
        x[feat_name] = mm_x.fit_transform(x[feat_name])

        if algorithm_.label_sheet:
            y = intermediate_data_handle['labels']
            mm_y = MinMaxScaler()
            y[[columns.get(is_label=True).name]] = mm_y.fit_transform(y[[columns.get(is_label=True).name]])
            intermediate_paper_handle = ContentFile(pickle.dumps([mm_x, mm_y]))
        else:
            intermediate_paper_handle = ContentFile(pickle.dumps(mm_x))
        new_paper = Paper(user=req.user, role=3, name=f"Pre-processing Time Series #{algorithm_.id} Normalizer")
        new_paper.file.save(f"ts_{algorithm_.id}_normalizer.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.normalizers = new_paper

        x_code = columns.get(belong_time_series=True, is_index=True).name
        if algorithm_.label_sheet:
            y_code = columns.get(belong_time_series=False, is_index=True).name
            code_list = list_union(np.unique(x[x_code].values), np.unique(y[y_code].values))
        else:
            code_list = np.unique(x[x_code].values)
        n_code = len(code_list)
        code_dict = dict(zip(code_list, range(n_code)))

        start_date = pd.Timestamp(algorithm_.from_datetime)
        end_date = pd.Timestamp(algorithm_.to_datetime)
        if start_date >= end_date:
            raise Exception('Start datetime must be earlier than end datetime.')
        scale_date = np.linspace(start_date.value, end_date.value, algorithm_.periods + 1)
        scale_date = pd.to_datetime(scale_date)
        date_variable_name = columns.get(belong_time_series=True, is_date=True).name
        date_slicer = pd.cut(x[date_variable_name].values, scale_date, ordered=True)
        x[date_variable_name] = date_slicer.codes.astype('int32')

        feat_avg = np.nanmean(x[feat_name], axis=0)
        attributes = np.empty(shape=(n_code, algorithm_.periods + 1, len(feat_name)), dtype=np.float32)
        for (code, week), transaction in x.groupby([x_code, date_variable_name]):
            if code not in code_list:
                continue
            for i in range(len(feat_name)):
                name = feat_name[i]
                if columns.get(name=name).log:
                    attributes[code_dict[code], week + 1, i] = np.log(np.nansum(np.exp(transaction[name]) - 1) + 1)
                else:
                    attributes[code_dict[code], week + 1, i] = np.nanmean(transaction[name])
        for i in range(len(feat_name)):
            name = feat_name[i]
            if columns.get(name=name).fill_na_avg:
                attributes[:, :, i] = np.nan_to_num(attributes[:, :, i], nan=feat_avg[i], posinf=feat_avg[i],
                                                    neginf=feat_avg[i])
            else:
                attributes[:, :, i] = np.nan_to_num(attributes[:, :, i], nan=0, posinf=0, neginf=0)

        if algorithm_.label_sheet:
            score = np.full(n_code, 0, dtype=np.float32)
            for code, score_array in y.groupby(y_code):
                if code not in code_list:
                    continue
                score[code_dict[code]] = np.nanmean(score_array[columns.get(is_label=True).name])
            intermediate_paper_handle = ContentFile(
                pickle.dumps({'index': code_list, 'X': attributes, 'Y': score, 'columns': feat_name})
            )
        else:
            intermediate_paper_handle = ContentFile(
                pickle.dumps({'index': code_list, 'X': attributes, 'columns': feat_name})
            )
        new_paper = Paper(user=req.user, role=2, name=f"Pre-processing Time Series #{algorithm_.id} Transformed")
        new_paper.file.save(f"ts_{algorithm_.id}_matrix.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.matrix = new_paper

        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        algorithm_.step.status = 4
        algorithm_.step.save()
        algorithm_.error_message = str(e)
        algorithm_.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_ts/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    algorithm_.save()
    algorithm_.step.status = 3
    algorithm_.step.save()
    context = {"color": "success", "content": "The columns are successfully configured.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)
