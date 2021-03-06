# Generated by Django 3.2.4 on 2021-09-16 02:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0003_alter_paper_name'),
        ('task_manager', '0002_auto_20210724_0228'),
    ]

    operations = [
        migrations.AddField(
            model_name='step',
            name='dataframe',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='dataframe', to='library.paper'),
        ),
        migrations.AddField(
            model_name='step',
            name='error_message',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='step',
            name='linked_data',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='linked_data', to='library.paper'),
        ),
        migrations.AddField(
            model_name='step',
            name='note',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='step',
            name='predicted_data',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='predicted_data', to='library.paper'),
        ),
    ]
