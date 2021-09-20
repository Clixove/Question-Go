import pickle
import typing

import numpy as np
import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.core.validators import MinValueValidator
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.preprocessing import MinMaxScaler

import task_manager.views
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


class DataPicker(forms.Form):
    step = forms.ModelChoiceField(Step.objects.all(), widget=forms.HiddenInput())
    paper = forms.ModelChoiceField(Paper.objects.all(), widget=forms.Select({'class': 'form-select'}), empty_label=None)

    def load_choices(self, user, search):
        queryset = Paper.objects.filter(user=user, name__contains=search)
        self.fields['step'].queryset = Step.objects.filter(task__user=user, status__in=[1, 3, 4])
        self.fields['paper'].queryset = queryset


@permission_required("pre_time_series.change_timeseries")
@csrf_exempt
@require_POST
def import_data(req):
    # re-write task_manager.views.use_data
    data_picker = DataPicker(req.POST)
    data_picker.load_choices(req.user, str())
    if not data_picker.is_valid():
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    step = data_picker.cleaned_data['step']
    paper = data_picker.cleaned_data['paper']
    step.linked_data = paper
    step.save()
    # ------------------------------------

    algorithm_ = TimeSeries.objects.get(step=step)
    for sheet in Sheet.objects.filter(algorithm=algorithm_):
        sheet.delete()
    try:
        sheet_names = pd.ExcelFile(step.linked_data.file.path).sheet_names
    except Exception as e:
        context = {"color": "success", "content": f"Cannot discover sheet in this Microsoft Excel file. {e}",
                   "refresh": f"/pre_ts/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    for sheet in sheet_names:
        new_sheet = Sheet(algorithm=algorithm_, name=sheet)
        new_sheet.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


class SelectSheet(PublicAlgorithm):
    time_series_sheet = forms.ModelChoiceField(
        Sheet.objects.all(), empty_label=None,
        widget=forms.Select({"class": "form-select"}),
        help_text="The sheet containing multi-dimensional time series, usually expanded as a 2D table. "
                  "Sample index and time points are two of columns.",
    )
    labels_sheet = forms.ModelChoiceField(
        Sheet.objects.all(),
        widget=forms.Select({"class": "form-select"}), required=False,
        help_text="The classification or regression label corresponding to time series.")

    def load_choices(self, algorithm):
        self.fields['time_series_sheet'].queryset = \
            self.fields['labels_sheet'].queryset = Sheet.objects.filter(algorithm=algorithm)


@csrf_exempt
@require_POST
@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def select_sheet(req):
    ss = SelectSheet(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = ss.is_valid()
    algorithm_ = ss.cleaned_data['algorithm']
    step = algorithm_.step
    v2 = step.open_permission(req.user)
    ss.load_choices(algorithm_)
    v3 = ss.is_valid()
    if not (v1 and v2 and v3):
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
        for sheet in Sheet.objects.filter(algorithm=algorithm_):
            sheet.is_time_series = sheet == ss.cleaned_data['time_series_sheet']
            sheet.is_label = sheet == ss.cleaned_data['labels_sheet']
            sheet.save()
        [c.delete() for c in Column.objects.filter(algorithm=algorithm_)]
        ts_sheet = pd.read_excel(step.linked_data.file.path, sheet_name=ss.cleaned_data['time_series_sheet'].name)
        for column in ts_sheet.columns:
            new_column = Column(algorithm=algorithm_, name=column)
            new_column.save()
        if ss.cleaned_data['labels_sheet']:
            label_sheet = pd.read_excel(step.linked_data.file.path, sheet_name=ss.cleaned_data['labels_sheet'].name)
            for column in label_sheet.columns:
                new_column = Column(algorithm=algorithm_, name=column, belong_time_series=False)
                new_column.save()
        else:
            label_sheet = None
        intermediate_paper_handle = ContentFile(pickle.dumps({'time_series': ts_sheet, 'labels': label_sheet}))
        new_paper = Paper(user=req.user, role=2, name=f"Pre-processing Time Series #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"pre_ts_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.dataframe = new_paper
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_ts/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The sheets are successfully assigned.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_sheet(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.dataframe = None
    algorithm_.save()
    [column.delete() for column in Column.objects.filter(algorithm=algorithm_)]
    algorithm_.step.status = 1
    algorithm_.step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


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

    def load_choices(self, algorithm):
        if algorithm.sheet_set.filter(is_label=True).exists():
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
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    select_sheet_ = SelectSheet()
    if Sheet.objects.filter(algorithm=algorithm_).exists():
        select_sheet_.initial = {
            'time_series_sheet': Sheet.objects.filter(algorithm=algorithm_, is_time_series=True).first(),
            'labels_sheet': Sheet.objects.filter(algorithm=algorithm_, is_label=True).first(),
        }
    select_sheet_.link_to_algorithm(algo_id)
    select_sheet_.load_choices(algorithm_)
    ac = SelectColumns(initial={
        'from_datetime': algorithm_.from_datetime, 'to_datetime': algorithm_.to_datetime,
        'periods': algorithm_.periods,
    })
    ac.link_to_algorithm(algo_id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/pre_ts/import',
        'select_sheet': select_sheet_,
        'time_series_sheet': algorithm_.sheet_set.filter(is_time_series=True).first(),
        'labels_sheet': algorithm_.sheet_set.filter(is_label=True).first(),
        'assign_column': ac,
        'time_series_columns': Column.objects.filter(algorithm=algorithm_, belong_time_series=True),
        'labels_columns': Column.objects.filter(algorithm=algorithm_, belong_time_series=False)
    }
    return render(req, "pre_time_series/main.html", context)


@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def clear_transform(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = TimeSeries.objects.get(id=algo_id)
    except TimeSeries.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    step = algorithm_.step
    if step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.normalizers = None
    algorithm_.save()
    step.predicted_data = None
    step.status = 1
    step.save()
    return redirect(f"/pre_ts/{algorithm_.id}")


@csrf_exempt
@require_POST
@permission_required("pre_time_series.change_timeseries",
                     login_url="/task/retrieve?message=You don't have permission to change this algorithm.&color=danger")
def transform(req):
    sc = SelectColumns(req.POST)
    # ---------- Algorithm Ownership Validator v2 START ----------
    v1 = sc.is_valid()
    algorithm_ = sc.cleaned_data['algorithm']
    step = algorithm_.step
    v2 = algorithm_.step.open_permission(req.user)
    sc.load_choices(algorithm_)
    v3 = sc.is_valid()
    if not (v1 and v2 and v3):
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
        columns = algorithm_.column_set.all()
        for column in columns:
            column.is_date = column == sc.cleaned_data['date']
            column.is_index = (column == sc.cleaned_data['company_trans']) or \
                              (column == sc.cleaned_data['company_score'])
            column.is_label = column == sc.cleaned_data['score']
            column.use = column in sc.cleaned_data['use']
            column.log = column in sc.cleaned_data['log']
            column.diff = column in sc.cleaned_data['diff']
            column.fill_na_avg = column in sc.cleaned_data['fill_na_avg']
            column.save()
        algorithm_.from_datetime = sc.cleaned_data['from_datetime']
        algorithm_.to_datetime = sc.cleaned_data['to_datetime']
        algorithm_.periods = sc.cleaned_data['periods']

        with open(algorithm_.dataframe.file.path, 'rb') as f:
            intermediate_data_handle = pickle.load(f)
        x = intermediate_data_handle['time_series']

        features_log = [z.name for z in columns.filter(log=True)]
        features_use = [z.name for z in columns.filter(use=True)]
        x[features_log] = x[features_log].apply(lambda z: np.log(z + 1))
        mm_x = MinMaxScaler()
        x[features_use] = mm_x.fit_transform(x[features_use])

        if algorithm_.sheet_set.filter(is_label=True).exists():
            y = intermediate_data_handle['labels']
            mm_y = MinMaxScaler()
            y[[columns.get(is_label=True).name]] = mm_y.fit_transform(y[[columns.get(is_label=True).name]])
            intermediate_paper_handle = ContentFile(pickle.dumps([mm_x, mm_y]))
        else:
            intermediate_paper_handle = ContentFile(pickle.dumps(mm_x))
        new_paper = Paper(user=req.user, role=3, name=f"Pre-processing Time Series #{algorithm_.id} Normalizer")
        new_paper.file.save(f"pre_ts_{algorithm_.id}_normalizer.pkl", intermediate_paper_handle)
        new_paper.save()
        algorithm_.normalizers = new_paper
        algorithm_.save()

        x_code = columns.get(belong_time_series=True, is_index=True).name
        if algorithm_.sheet_set.filter(is_label=True).exists():
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

        feat_avg = np.nanmean(x[features_use], axis=0)
        attributes = np.empty(shape=(n_code, algorithm_.periods + 1, len(features_use)), dtype=np.float32)
        for (code, week), transaction in x.groupby([x_code, date_variable_name]):
            if code not in code_list:
                continue
            for i in range(len(features_use)):
                name = features_use[i]
                if columns.get(name=name).log:
                    attributes[code_dict[code], week + 1, i] = np.log(np.nansum(np.exp(transaction[name]) - 1) + 1)
                else:
                    attributes[code_dict[code], week + 1, i] = np.nanmean(transaction[name])
        for i in range(len(features_use)):
            name = features_use[i]
            if columns.get(name=name).fill_na_avg:
                attributes[:, :, i] = np.nan_to_num(attributes[:, :, i], nan=feat_avg[i], posinf=feat_avg[i],
                                                    neginf=feat_avg[i])
            else:
                attributes[:, :, i] = np.nan_to_num(attributes[:, :, i], nan=0, posinf=0, neginf=0)

        if algorithm_.sheet_set.filter(is_label=True).exists():
            score = np.full(n_code, 0, dtype=np.float32)
            for code, score_array in y.groupby(y_code):
                if code not in code_list:
                    continue
                score[code_dict[code]] = np.nanmean(score_array[columns.get(is_label=True).name])
            intermediate_paper_handle = ContentFile(
                pickle.dumps({'index': code_list, 'X': attributes, 'Y': score, 'columns': features_use})
            )
        else:
            intermediate_paper_handle = ContentFile(
                pickle.dumps({'index': code_list, 'X': attributes, 'columns': features_use})
            )
        new_paper = Paper(user=req.user, role=2, name=f"Pre-processing Time Series #{algorithm_.id} Transformed")
        new_paper.file.save(f"pre_ts_{algorithm_.id}_transformed.pkl", intermediate_paper_handle)
        new_paper.save()
        step.predicted_data = new_paper
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_ts/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The columns are successfully configured.",
               "refresh": f"/pre_ts/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)
