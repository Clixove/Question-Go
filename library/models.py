from os.path import join

from django.conf.global_settings import MEDIA_ROOT
from django.contrib.auth.models import Group, User
from django.db import models


class GroupStorage(models.Model):
    group = models.OneToOneField(Group, on_delete=models.CASCADE)
    user_init_storage = models.IntegerField(verbose_name="Group Storage (MiB)", default=0)

    def __str__(self):
        return self.group.name


class UserStorage(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    specific_storage = models.IntegerField(verbose_name="User Storage (MiB)", default=0)

    def total_storage_bytes(self):
        storages = GroupStorage.objects.filter(group__in=self.user.groups.all())
        storage_mb = sum([x.user_init_storage for x in storages]) + self.specific_storage
        return storage_mb * 1024 ** 2

    def __str__(self):
        return self.user.username

    def upload_permission(self, new_file):
        return self.used_storage_bytes() + new_file.size <= self.total_storage_bytes()

    def used_storage_bytes(self):
        users_papers = Paper.objects.filter(user=self.user)
        return sum([x.file.size for x in users_papers])


class Paper(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    modified_time = models.DateTimeField(auto_now=True)
    role = models.IntegerField(choices=[(1, "Data"), (2, "Intermediate"), (3, "Model"), (4, "Result")], default=1)
    name = models.CharField(max_length=256)
    file = models.FileField(upload_to=join(MEDIA_ROOT, 'papers/'))

    def __str__(self):
        return self.name
