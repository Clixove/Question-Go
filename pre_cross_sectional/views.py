import pickle

import numpy as np
import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder

import task_manager.views
from task_manager.models import OpenedTask
from .models import *
from .safe_math import safe_eval


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(PreProcessing.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = PreProcessing.objects.get(id=algo_id)


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
        task=opened_task, name="Pre-processing Cross-sectional Data",
        view_link=f"/pre_cross_sectional/{new_algorithm.id}", model_id=new_algorithm.id
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
    step = csp.step
    v2 = step.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    if step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    if not step.predicted_data:
        context = {"color": "danger", "content": "Please parse a dataset first."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        dataframe = pd.read_pickle(step.predicted_data.file.path)
        profile = ProfileReport(dataframe, title=f"Pre-processing Cross-sectional Data #{csp.id}",
                                plot={"dpi": 200, "image_format": "png"})
        csp.report = profile.to_html()
        csp.save()
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
    step.status = 3
    step.save()
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
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    return HttpResponse(algorithm_.report)


@permission_required("pre_cross_sectional.change_preprocessing")
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
    algorithm_ = PreProcessing.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2, name=f"Cross-sectional Data Pre-processing #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"csp_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
        new_paper.save()
        # step.linked_data = new_paper
        step.predicted_data = new_paper
        step.save()
        for column in Column.objects.filter(algorithm=algorithm_):
            column.delete()
        for col in table.columns:
            new_column = Column(algorithm=algorithm_, name=col)
            new_column.save()
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/pre_cross_sectional/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Asynchronous Algorithm START   ----------
    algorithm_.save()
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/pre_cross_sectional/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


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


def pandas_drop_column(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.drop(
        columns=[x.name for x in config.cleaned_data['targeted_columns']],
        inplace=True,
    )
    config.cleaned_data['targeted_columns'].delete()
    return dataframe


class FillNa(PublicPreProcessing):
    method = forms.ChoiceField(
        choices=(
            (None, "use constant to fill gap (eg. 0)"),
            ("ffill", "propagate last valid observation forward to next valid"),
            ("bfill", "use next valid observation to fill gap"),
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
    constant = forms.FloatField(widget=forms.NumberInput({"class": "form-control"}), required=False)


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
                col: np.nanmean(np.sort(dataframe[col])[round(.05 * n):round(.95 * n)])
                for col in columns
            }
        elif config.cleaned_data['quick_constant'] == 'mode':
            na_value = {}
            for col in columns:
                not_nan_array = dataframe[col].dropna().values
                not_nan_array_uni, not_nan_array_freq = np.unique(not_nan_array, return_counts=True)
                na_value[col] = not_nan_array_uni[np.argmax(not_nan_array_freq)]
        elif config.cleaned_data['quick_constant'] == 'min':
            na_value = dataframe[columns].min(axis=0).to_dict()
        elif config.cleaned_data['quick_constant'] == 'max':
            na_value = dataframe[columns].max(axis=0).to_dict()
        else:
            na_value = config.cleaned_data['constant'] or 0
        dataframe[columns] = dataframe[columns].fillna(value=na_value)
    return dataframe


class Cast(PublicPreProcessing):
    data_type = forms.ChoiceField(
        choices=(
            ('numerical', 'Numerical (Automatically decided)'),
            ('datetime', 'Datetime'),
            ('timedelta', 'Datetime duration'),
            ('string', 'String'),
            ('int32', 'Integer 32-bit'),
            ('int64', 'Integer 64-bit'),
            ('float32', 'Float 32-bit'),
            ('float64', 'Float 64-bit'),
        ),
        widget=forms.Select({"class": "form-select"}),
    )
    datetime_format = forms.CharField(
        widget=forms.TextInput({"class": "form-control"}),
        required=False,
        help_text="(eg. %Y-%m-%d) <a "
                  "href='https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior'>"
                  "What is datetime format?</a>",
    )
    datetime_combined_from_multiple_columns = forms.BooleanField(
        required=False, initial=False,
        widget=forms.Select({"class": "form-select"}, choices=((True, 'Yes'), (False, 'No'))),
        help_text="The keys can be common abbreviations like [‘year’, ‘month’, ‘day’, ‘minute’, ‘second’, ‘ms’, "
                  "‘us’, ‘ns’]) or plurals of the same.",
    )
    datetime_duration_unit = forms.ChoiceField(
        choices=(
            ('W', 'Week'), ('D', 'Day'), ('h', 'Hour'), ('m', 'Minute'), ('S', 'Second'),
            ('us', 'Micro-second'), ('ns', 'Nano-second')
        ),
        initial='D',
        widget=forms.Select({"class": "form-select"})
    )


def pandas_cast(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [x.name for x in config.cleaned_data['targeted_columns']]
    if config.cleaned_data['data_type'] == 'numerical':
        for col in columns:
            dataframe[col] = pd.to_numeric(dataframe[col])
    elif config.cleaned_data['data_type'] == 'datetime':
        if config.cleaned_data['datetime_combined_from_multiple_columns']:
            dataframe = pd.to_datetime \
                (dataframe, format=config.cleaned_data['datetime_format'], infer_datetime_format=True)
        else:
            for col in columns:
                dataframe[col] = pd.to_datetime \
                    (dataframe[col], format=config.cleaned_data['datetime_format'], infer_datetime_format=True)
    elif config.cleaned_data['data_type'] == 'timedelta':
        for col in columns:
            dataframe[col] = pd.to_timedelta(dataframe[col], unit=config.cleaned_data['datetime_duration_unit'])
    else:
        dataframe = dataframe.astype({col: config.cleaned_data['data_type'] for col in columns})
    return dataframe


class Encode(PublicPreProcessing):
    method = forms.ChoiceField(
        choices=(('o', 'one-hot encode'), ('t', 'target encode')),
        widget=forms.Select({"class": "form-select"}),
        initial='t'
    )


def sklearn_encode(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [x.name for x in config.cleaned_data['targeted_columns']]
    if config.cleaned_data['method'] == 'o':
        this_algorithm = config.cleaned_data['targeted_columns'].first().algorithm
        oh = OneHotEncoder()
        for col in columns:
            oh_matrix = oh.fit_transform(dataframe[[col]])
            categories = [f"{col}/{col_}" for col_ in oh.categories_[0]]
            oh_frame = pd.DataFrame(data=oh_matrix.toarray(), columns=categories)
            dataframe = pd.concat([dataframe, oh_frame], axis=1, join='inner')
            for col_ in categories:
                new_column = Column(algorithm=this_algorithm, name=col_)
                new_column.save()
    elif config.cleaned_data['method'] == 't':
        for col in columns:
            content, frequency = np.unique(dataframe[col], return_counts=True)
            frequency = frequency / max(dataframe.shape[0], 1)
            content = dict(zip(content, frequency))
            dataframe[col] = dataframe[col].apply(lambda x: content[x], convert_dtype='float32')
    return dataframe


class MathOp(PublicPreProcessing):
    new_name = forms.CharField(
        widget=forms.TextInput({'class': 'form-control'}),
        max_length=99,
        help_text="The name of newly constructed variable."
    )
    expression = forms.CharField(
        widget=forms.TextInput({"class": "form-control"}),
        max_length=99,
        help_text="'x' is variables as column arrays. Simple mathematical operations only."
    )


def math_op(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [x.name for x in config.cleaned_data['targeted_columns']]
    dataframe[config.cleaned_data['new_name']] = dataframe[columns].apply(
        lambda x: safe_eval(config.cleaned_data['expression'], x.values))
    new_column = Column(algorithm=config.cleaned_data['algorithm'], name=config.cleaned_data['new_name'])
    new_column.save()
    return dataframe


preprocessing_wrapper_menu = {
    "drop_column": {"form": DropColumns, "function": pandas_drop_column},
    "fill_na": {"form": FillNa, "function": pandas_fill_na},
    "cast": {"form": Cast, "function": pandas_cast},
    "encode": {"form": Encode, "function": sklearn_encode},
    "math_op": {"form": MathOp, "function": math_op},
}


@permission_required("pre_cross_sectional.view_preprocessing",
                     login_url="/task/retrieve?message=You don't have permission to view this algorithm.&color=danger")
def view_csp(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = PreProcessing.objects.get(id=algo_id)
    except PreProcessing.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    profile_sheet = Profile()
    profile_sheet.link_to_algorithm(algo_id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/pre_cross_sectional/import',
        "profile": profile_sheet,
    }
    for form_name, form_config in preprocessing_wrapper_menu.items():
        preprocessing_sheet = preprocessing_wrapper_menu[form_name]['form']()
        preprocessing_sheet.link_to_algorithm(algo_id)
        preprocessing_sheet.fields['targeted_columns'].widget.__dict__['attrs']['id'] = form_name + '_tc'
        preprocessing_sheet.fields['algorithm'].widget.__dict__['attrs']['id'] = form_name + '_algo'
        context[form_name] = preprocessing_sheet
    return render(req, "pre_cross_sectional/main.html", context)


@permission_required("pre_cross_sectional.change_preprocessing")
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
    step = csp.step
    v2 = step.open_permission(req.user)
    if not (v1 and v2):
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    # ---------- Algorithm Ownership Validator v2 END   ----------
    step = csp.step
    if not step.predicted_data:
        context = {"color": "danger", "content": "This instance doesn't contain data and variables."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        dataframe = pd.read_pickle(step.predicted_data.file.path)
        dataframe = preprocessing_wrapper_menu[form_name]['function'](preprocessing_form, dataframe)
        dataframe.to_pickle(step.predicted_data.file.path)
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.", "refresh": f"/pre_cross_sectional/{csp.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The dataset has been updated.",
               "refresh": f"/pre_cross_sectional/{csp.id}"}
    return render(req, "task_manager/hint_widget.html", context)
