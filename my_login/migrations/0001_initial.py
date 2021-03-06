# Generated by Django 4.0.4 on 2022-04-15 02:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='InvitationCode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254)),
                ('invitation_code', models.TextField(unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='RegistryEntries',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('registry_name', models.CharField(max_length=64, unique=True)),
                ('groups', models.ManyToManyField(to='auth.group')),
            ],
        ),
    ]
