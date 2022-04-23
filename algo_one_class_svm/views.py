import io
import json
import pickle

import numpy as np
import pandas as pd
from django import forms
from django.contrib.auth.decorators import permission_required
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.utils.datastructures import MultiValueDictKeyError
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import OneClassSVM

import task_manager.views
from task_manager.models import OpenedTask
from .models import *


class PublicAlgorithm(forms.Form):
    algorithm = forms.ModelChoiceField(MyOneClassSVM.objects.all(), widget=forms.HiddenInput())

    def link_to_algorithm(self, algo_id):
        self.fields['algorithm'].initial = MyOneClassSVM.objects.get(id=algo_id)


class VariablePicker(PublicAlgorithm):
    Independent_Variables_X = forms.ModelMultipleChoiceField(
        Column.objects.all(), widget=forms.CheckboxSelectMultiple()
    )
    Dependent_Variable_Y = forms.ModelChoiceField(
        Column.objects.all(), widget=forms.Select({"class": "form-select"}), initial=""
    )

    def load_choices(self, algorithm):
        columns = Column.objects.filter(algorithm=algorithm)
        self.fields['Independent_Variables_X'].queryset = columns
        self.fields['Dependent_Variable_Y'].queryset = columns
        self.fields['Dependent_Variable_Y'].initial = columns.last()


class Train(PublicAlgorithm):
    running_mode = forms.ChoiceField(
        widget=forms.Select({"class": "form-select"}),
        choices=[("5_fold", "5 fold cross validation"),
                 ("split", "Random split to 80% training set, 20% validation set"),
                 ("full_train", "Applying all samples for training")],
    )
    random_seed = forms.IntegerField(
        min_value=1, max_value=9999999, required=False, widget=forms.NumberInput({"class": "form-control"}),
        help_text="Not required. From 1 to 9999999, leave blank if not purpose to fix."
    )
    abnormal_class_name = forms.ChoiceField(
        widget=forms.Select({'class': 'form-control form-select'}),
        help_text='Name of the class, samples belong to which is taken as abnormal. Normal classes is converted to '
                  '1, and the abnormal class is converted to -1.'
    )
    nu = forms.FloatField(
        min_value=0, max_value=1, widget=forms.NumberInput({'class': 'form-control'}), initial=0.5,
        help_text='An upper bound on the fraction of training errors and a lower bound of the fraction of support '
                  'vectors. It\'s in interval (0, 1].'
    )
    kernel = forms.ChoiceField(
        widget=forms.Select({'class': 'form-control'}),
        help_text='The kernel function of SVM.',
        choices=[('linear', 'Linear Function'), ('poly', 'Polynomial'), ('rbf', 'Radial Basis Function'),
                 ('sigmoid', 'Sigmoid Function')],
        initial='rbf',
    )
    degree = forms.IntegerField(
        widget=forms.NumberInput({'class': 'form-control'}),
        help_text='Degree of the polynomial kernel function. Required only when the kernel function is polynomial.'
                  '(Integer no smaller than 2)',
        required=False, min_value=2,
    )


@permission_required(
    'algo_one_class_svm.add_myoneclasssvm',
    login_url='/task/retrieve?message=You don\'t have access to this algorithm.&color=danger'
)
def add_one_class_svm(req):
    try:
        opened_task = OpenedTask.objects.get(user=req.user).task
    except OpenedTask.DoesNotExist:
        return redirect("/task/instances?message=You should open a task first.&color=danger")
    new_algorithm = MyOneClassSVM()
    new_algorithm.save()
    new_step = Step(
        task=opened_task, name="One-class SVM", view_link=f"/algo_one_class_svm/{new_algorithm.id}",
        model_id=new_algorithm.id
    )
    new_step.save()
    new_algorithm.step = new_step
    new_algorithm.save()
    return redirect(new_step.view_link)


@permission_required("algo_one_class_svm.view_myoneclasssvm",
                     login_url="/task/retrieve?message=You don't have access to view algorithms.&color=danger")
def view_one_class_svm(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyOneClassSVM.objects.get(id=algo_id)
    except MyOneClassSVM.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    variable_picker = VariablePicker()
    variable_picker.link_to_algorithm(algorithm_.id)
    variable_picker.load_choices(algorithm_)
    x_var = algorithm_.column_set.filter(x_column=True)
    y_var = algorithm_.column_set.filter(y_column=True)
    train_config = Train()
    train_config.link_to_algorithm(algorithm_.id)
    context = {
        "algorithm": algorithm_, "note": task_manager.views.display_note(algorithm_.step),
        "search_data": task_manager.views.display_data_picker(algorithm_.step),
        "import_data_target": '/algo_one_class_svm/import',
        "predict_data_target": '/algo_one_class_svm/predict',
        "variable_picker": variable_picker, "x_var": x_var, "y_var": y_var,
    }
    try:  # MUST NOT SHARE THE CASE WITH CLASSIFICATION EVALUATION
        train_config.fields['abnormal_class_name'].choices = [(x, x) for x in json.loads(algorithm_.class_list)]
        context['train_config'] = train_config
    except json.JSONDecodeError:
        pass
    try:  # TRADE-OFF BETWEEN TRY-EXCEPT AND IF-ELSE
        if algorithm_.mode == '5_fold':
            context['evaluate'] = zip(json.loads(algorithm_.confusion_matrix), json.loads(algorithm_.hyper_parameters))
        else:
            context['c_mat'] = json.loads(algorithm_.confusion_matrix)
            context['h_para'] = json.loads(algorithm_.hyper_parameters)
    except json.JSONDecodeError:
        pass
    return render(req, "algo_one_class_svm/main.html", context)


@permission_required("algo_one_class_svm.change_myoneclasssvm")
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
    algorithm_ = MyOneClassSVM.objects.get(step=step)
    try:
        # ---------- Asynchronous Algorithm START   ----------
        intermediate_paper_handle = ContentFile(pickle.dumps(table))
        new_paper = Paper(user=req.user, role=2,
                          name=f"One-class SVM #{algorithm_.id} Parsed Data")
        new_paper.file.save(f"one_class_svm_{algorithm_.id}_parsed_data.pkl", intermediate_paper_handle)
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
                   "refresh": f"/algo_one_class_svm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The file is successfully parsed.",
               "refresh": f"/algo_one_class_svm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_one_class_svm.change_myoneclasssvm")
@csrf_exempt
@require_POST
def set_variables(req):  # SPECIFIED
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
    for column in variable_picker.cleaned_data['Independent_Variables_X']:
        column.x_column = True
        column.save()
    column = variable_picker.cleaned_data['Dependent_Variable_Y']
    try:
        data = pd.read_pickle(algorithm_.dataframe.file.path)
        data = data[column.name]
        algorithm_.class_list = json.dumps(np.unique(data).tolist())
        algorithm_.save()
    except Exception as e:
        context = {"color": "danger", "content": f"The dependent variable cannot be parsed. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    column.y_column = True
    column.save()
    context = {"color": "success", "content": "Set variables successfully.",
               "refresh": f"/algo_one_class_svm/{algorithm_.id}"}  # REQUIRED FOR REFRESHING
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_one_class_svm.change_myoneclasssvm",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_variables(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyOneClassSVM.objects.get(id=algo_id)
    except MyOneClassSVM.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    for column in Column.objects.filter(algorithm=algorithm_):
        column.x_column = column.y_column = False
        column.save()
    algorithm_.class_list = str()
    algorithm_.save()
    return redirect(f"/algo_one_class_svm/{algorithm_.id}")


@permission_required("algo_one_class_svm.change_myoneclasssvm")
@csrf_exempt
@require_POST
def train_model(req):
    try:
        algorithm_ = MyOneClassSVM.objects.get(id=req.POST['algorithm'])
    except (MyOneClassSVM.DoesNotExist, MultiValueDictKeyError):
        context = {"color": "danger", "content": "The algorithm does not exist."}
        return render(req, "task_manager/hint_widget.html", context)
    train = Train(req.POST)
    train.fields['abnormal_class_name'].choices = [(x, x) for x in json.loads(algorithm_.class_list)]
    step = algorithm_.step
    if (not train.is_valid()) or (not step.open_permission(req.user)) or train.cleaned_data['nu'] == 0:
        context = {"color": "danger", "content": "Submission is not valid."}
        return render(req, "task_manager/hint_widget.html", context)
    if step.status == 2:
        context = {"color": "warning", "content": "Cannot start because this algorithm is busy."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        # ---------- Asynchronous Algorithm START ----------
        with open(algorithm_.dataframe.file.path, "rb") as f:
            dataframe = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        mode = train.cleaned_data['running_mode']
        algorithm_.abnormal_class_name = train.cleaned_data['abnormal_class_name']
        x, y = dataframe[x_col].values, dataframe[y_col].values
        y = np.where(y == y.dtype.type(train.cleaned_data['abnormal_class_name']), -1, 1)

        if mode == "5_fold":
            k_fold = KFold(n_splits=5, random_state=train.cleaned_data['random_seed'], shuffle=True)
            models_, confusion_matrix_list, hyper_parameters_list, support_vectors_list = [], [], [], []

            for (train_index, valid_index), k in zip(k_fold.split(x), range(5)):
                x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
                x_train = x_train[y_train]
                if train.cleaned_data['degree']:
                    mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'],
                                      degree=train.cleaned_data['degree'], max_iter=5000)
                else:
                    mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'], max_iter=5000)
                mdl.fit(x_train)
                models_.append(mdl)
                y_valid_hat = mdl.predict(x_valid)
                c1, c2 = y_valid == 1, y_valid_hat == 1
                c_mat = [[np.sum(~c1 & ~c2).__int__(), np.sum(~c1 & c2).__int__()],
                         [np.sum(c1 & ~c2).__int__(), np.sum(c1 & c2).__int__()]]
                confusion_matrix_list.append(c_mat)
                hyper_parameters_list.append({'degree': mdl.degree, 'kernel': mdl.kernel, 'nu': mdl.nu})
                support_vectors_list.append(mdl.support_vectors_)

            algorithm_.confusion_matrix = json.dumps(confusion_matrix_list)
            algorithm_.hyper_parameters = json.dumps(hyper_parameters_list, ensure_ascii=False)

            intermediate_paper_handle = ContentFile(pickle.dumps(models_))
            new_paper = Paper(user=req.user, role=3, name=f'One-class SVM #{algorithm_.id} Model')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper

            intermediate_paper_handle = ContentFile(pickle.dumps(support_vectors_list))
            new_paper = Paper(user=req.user, role=2, name=f'One-class SVM #{algorithm_.id} Support Vector')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_support_vector.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.support_vectors = new_paper

        elif mode == "split":
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, shuffle=True,
                                                                  random_state=train.cleaned_data['random_seed'])
            x_train = x_train[y_train]
            if train.cleaned_data['degree']:
                mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'],
                                  degree=train.cleaned_data['degree'], max_iter=5000)
            else:
                mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'], max_iter=5000)
            mdl.fit(x_train)
            y_valid_hat = mdl.predict(x_valid)
            c1, c2 = y_valid == 1, y_valid_hat == 1
            c_mat = [[np.sum(~c1 & ~c2).__int__(), np.sum(~c1 & c2).__int__()],
                     [np.sum(c1 & ~c2).__int__(), np.sum(c1 & c2).__int__()]]
            hyper_parameters = {'degree': mdl.degree, 'kernel': mdl.kernel, 'nu': mdl.nu}

            algorithm_.confusion_matrix = json.dumps(c_mat)
            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl.support_vectors_))
            new_paper = Paper(user=req.user, role=2, name=f'One-class SVM #{algorithm_.id} Support Vector')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_support_vector.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.support_vectors = new_paper

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'One-class SVM #{algorithm_.id} Model')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper

        else:  # mode == "full_train"
            if train.cleaned_data['degree']:
                mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'],
                                  degree=train.cleaned_data['degree'], max_iter=5000)
            else:
                mdl = OneClassSVM(kernel=train.cleaned_data['kernel'], nu=train.cleaned_data['nu'], max_iter=5000)
            x_train = x[y]
            mdl.fit(x_train)
            hyper_parameters = {'degree': mdl.degree, 'kernel': mdl.kernel, 'nu': mdl.nu}

            algorithm_.hyper_parameters = json.dumps(hyper_parameters, ensure_ascii=False)

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl.support_vectors_))
            new_paper = Paper(user=req.user, role=2, name=f'One-class SVM #{algorithm_.id} Support Vector')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_support_vector.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.support_vectors = new_paper

            intermediate_paper_handle = ContentFile(pickle.dumps(mdl))
            new_paper = Paper(user=req.user, role=3, name=f'One-class SVM #{algorithm_.id} Model')
            new_paper.file.save(f'one_class_svm_{algorithm_.id}_model.pkl', intermediate_paper_handle)
            new_paper.save()
            algorithm_.model = new_paper

        algorithm_.mode = mode
        algorithm_.save()
        # ---------- Asynchronous Algorithm END   ----------
    except Exception as e:
        step.status = 4
        step.error_message = str(e)
        step.save()
        context = {"color": "danger", "content": "Interrupted.",
                   "refresh": f"/algo_one_class_svm/{algorithm_.id}"}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 3
    step.save()
    context = {"color": "success", "content": "The model has been trained and evaluated.",
               "refresh": f"/algo_one_class_svm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context)


@permission_required("algo_one_class_svm.change_myoneclasssvm",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
def clear_model(req, algo_id):
    # ---------- Algorithm Ownership Navigator START ----------
    try:
        algorithm_ = MyOneClassSVM.objects.get(id=algo_id)
    except MyOneClassSVM.DoesNotExist:
        return redirect("/task/retrieve?message=This instance doesn't exist.&color=danger")
    if not algorithm_.step.open_permission(req.user):
        return redirect("/task/retrieve?message=You don't have access to this algorithm.&color=danger")
    # ---------- Algorithm Ownership Navigator END   ----------
    if algorithm_.step.status == 2:
        return redirect("/task/retrieve?message=Cannot start because this algorithm is busy.&color=warning")
    algorithm_.model = None
    algorithm_.mode = str()
    algorithm_.hyper_parameters = str()
    algorithm_.abnormal_class_name = str()
    algorithm_.confusion_matrix = str()
    algorithm_.support_vectors = None
    algorithm_.save()
    return redirect(f"/algo_one_class_svm/{algorithm_.id}")


def most_frequent_item(a: np.array):
    a = a.tolist()
    return max(set(a), key=a.count)


@permission_required("algo_one_class_svm.change_myoneclasssvm",
                     login_url="/task/retrieve?message=You don't have access to change algorithms.&color=danger")
@csrf_exempt
@require_POST
def predict(req):
    # ---------- Import Data Tool V2 START ----------
    table, step, error_message = task_manager.views.import_predicting_set_v2(req)
    if table is None:
        context = {'color': 'danger', 'content': error_message}
        return render(req, 'task_manager/hint_widget.html', context)
    # "step.status" has been changed to 2.
    # ---------- Import Data Tool V2 END   ----------
    algorithm_ = MyOneClassSVM.objects.get(step=step)
    if not algorithm_.model:
        context = {"color": "danger", "content": "This step doesn't have a trained model."}
        return render(req, "task_manager/hint_widget.html", context)
    step.status = 2
    step.save()
    try:
        with open(algorithm_.model.file.path, "rb") as f:
            model = pickle.load(f)
        x_col = [x.name for x in Column.objects.filter(algorithm=algorithm_, x_column=True)]
        y_col = Column.objects.filter(algorithm=algorithm_, y_column=True).first().name
        x = table[x_col].values
        if algorithm_.mode == '5_fold':
            y_hat = np.array([model[i].predict(x) for i in range(5)], dtype='object')
            y_hat = np.apply_along_axis(most_frequent_item, axis=0, arr=y_hat)
            table[y_col] = y_hat
        else:
            table[y_col] = model.predict(x)
        table_bin = io.BytesIO()
        with pd.ExcelWriter(table_bin) as f:
            table.to_excel(f, index=False)
    except Exception as e:
        step.status = 4
        step.save()
        context = {"color": "warning", "content": f"Interrupted. {e}"}
        return render(req, "task_manager/hint_widget.html", context)
    new_paper = Paper(user=req.user, role=4, name=f"One-class SVM #{algorithm_.id} Predict")
    new_paper.file.save(f"one_class_svm_{algorithm_.id}_predict.xlsx", table_bin)
    new_paper.save()
    step.predicted_data = new_paper
    step.status = 3
    step.save()
    context = {"color": "success", "content": "Prediction completed.",
               "refresh": f"/algo_one_class_svm/{algorithm_.id}"}
    return render(req, "task_manager/hint_widget.html", context=context)
