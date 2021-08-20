from django import forms
from django.contrib.auth.decorators import permission_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import FileResponse

import question_go_v2.settings
from .models import *
from os.path import basename


class PublicSelectMultiplePaper(forms.Form):
    def get_instances(self, user):
        self.fields['papers'] = forms.ModelMultipleChoiceField(Paper.objects.filter(user=user), required=False)
        return self


class AddPaper(forms.Form):
    role = forms.ChoiceField(widget=forms.Select({"class": "form-select"}),
                             choices=[(1, "Data"), (2, "Intermediate"), (3, "Model"), (4, "Result")])
    file = forms.FileField(widget=forms.FileInput({"class": "form-control"}))


class DeletePaper(PublicSelectMultiplePaper):
    pass


class RenamePaper(PublicSelectMultiplePaper):
    new_name = forms.CharField(max_length=256, widget=forms.TextInput({"class": "form-control"}), required=False,
                               help_text="Leave blank to revoke to original filename.")


class SearchPaper(forms.Form):
    search = forms.CharField(max_length=256, widget=forms.TextInput({"class": "form-control"}), required=False,
                             label="")


@permission_required("library.view_paper", login_url="/main?message=No permission to view papers.&color=danger")
def view_library(req):
    sp = SearchPaper(req.GET)
    papers = Paper.objects.filter(user=req.user, name__contains=sp.cleaned_data['search']) \
        if sp.is_valid() else Paper.objects.filter(user=req.user)
    my_storage, created = UserStorage.objects.get_or_create(user=req.user)
    if created:
        my_storage.save()
    total_storage = my_storage.total_storage_bytes()
    used_storage = my_storage.used_storage_bytes()
    context = {
        "TotalStorage": total_storage,
        "UsedStorage": used_storage,
        "RateStorage": 0 if total_storage == 0 else used_storage / total_storage * 100,
        "Papers": papers.order_by('role', '-modified_time'),
        "SearchSheet": SearchPaper(req.GET),
        "AddPaperSheet": AddPaper(),
        "timezone": question_go_v2.settings.TIME_ZONE,
        "RenamePaperSheet": RenamePaper(),
    }
    return render(req, "library/main.html", context)


@permission_required("library.view_paper", login_url="/library?message=No permission.&color=danger")
def view_paper(req, paper_id):
    try:
        paper = Paper.objects.get(id=paper_id, user=req.user)
    except Paper.DoesNotExist:
        return redirect("/library?message=This paper does not exist.&color=danger")
    return FileResponse(paper.file)


@permission_required("library.add_paper", login_url="/library?message=No permission.&color=danger")
@csrf_exempt
@require_POST
def add_paper(req):
    sheet = AddPaper(req.POST, req.FILES)
    if not sheet.is_valid():
        return redirect("/library?message=Submission is not valid.&color=danger")
    my_storage = UserStorage.objects.get(user=req.user)
    new_paper = Paper(user=req.user, file=sheet.cleaned_data['file'], role=sheet.cleaned_data['role'])
    new_paper.name = basename(new_paper.file.name)
    if not my_storage.upload_permission(new_paper.file):
        return redirect("/library?message=Your storage is used up.&color=danger")
    new_paper.save()
    return redirect("/library")


@csrf_exempt
@require_POST
@permission_required("library.delete_paper", login_url="/library?message=No permission to delete papers.&color=danger")
def delete_paper(req):
    dp = DeletePaper(req.POST)
    dp.get_instances(req.user)
    if not dp.is_valid():
        return redirect(f"/library?message=Submission is not valid.&color=danger")
    for paper in dp.cleaned_data['papers']:
        paper.delete()
    return redirect("/library")


@csrf_exempt
@require_POST
@permission_required("library.change_paper", login_url="/library?message=No permission to change papers.&color=danger")
def rename_paper(req):
    rp = RenamePaper(req.POST)
    rp.get_instances(req.user)
    if not rp.is_valid():
        return redirect(f"/library?message=Submission is not valid.&color=danger")
    for paper in rp.cleaned_data['papers']:
        if rp.cleaned_data['new_name']:
            paper.name = rp.cleaned_data['new_name']
        else:
            paper.name = basename(paper.file.path)
        paper.save()
    return redirect("/library")
