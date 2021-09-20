cd ~/question_go_v2/
conda activate webapp
python manage.py makemigrations
python manage.py migrate --fake pre_time_series zero
rm -r pre_time_series/migrations/__pycache__/
rm pre_time_series/migrations/0*
python manage.py makemigrations
python manage.py migrate --fake-initial