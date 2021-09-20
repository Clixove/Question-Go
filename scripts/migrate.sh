# shellcheck disable=SC2164
cd ~/question_go_v2/
conda activate webapp
python manage.py makemigrations
python manage.py migrate
