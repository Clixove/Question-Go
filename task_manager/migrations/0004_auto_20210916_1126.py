# Generated by Django 3.2.4 on 2021-09-16 03:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('task_manager', '0003_auto_20210916_1055'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='step',
            name='dataframe',
        ),
        migrations.AlterField(
            model_name='step',
            name='status',
            field=models.IntegerField(choices=[(1, 'PREPARED'), (2, 'RUNNING'), (3, 'DONE'), (4, 'INTERRUPTED')], default=1),
        ),
    ]
