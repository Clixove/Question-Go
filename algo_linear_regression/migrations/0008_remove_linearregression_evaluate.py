# Generated by Django 3.2.4 on 2021-07-24 01:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('algo_linear_regression', '0007_auto_20210721_1320'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='linearregression',
            name='evaluate',
        ),
    ]
