app_name = 'pre_time_series'
template = \
    f"""cd ~/question_go_v2/
conda activate webapp
python manage.py makemigrations
python manage.py migrate --fake {app_name} zero
rm -r {app_name}/migrations/__pycache__/
rm {app_name}/migrations/0*
python manage.py makemigrations
python manage.py migrate --fake-initial"""
with open(f"clean_migration_{app_name}.sh", "w") as f:
    f.write(template)
