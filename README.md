# QuestionGo
 Construct and reuse machine learning models

![](https://img.shields.io/badge/dependencies-Python%203.8--3.9-blue)
![](https://img.shields.io/badge/dependencies-Django%203.2-green)
![](https://img.shields.io/badge/tests-Chrome%2089--92%20%E2%9C%94-brightgreen)

## 1. Background

## 2. Functions

## 3. Acknowledge

The programming language is Python 3.9, dependent packages are listed in `requirements.txt`. Besides:

Pandas profiling: https://github.com/pandas-profiling/pandas-profiling

## 4. Design

### 4.1. Actions

### 4.2. Structure

## 5. Installation

The current folder of command line is the software's project root.

### 5.1. Build token files

Create `token` folder at project root. There should be several files in this folder:
- In `token/django_secret_key`, there should be a string about 52 characters, being a secret key for communication between client and web server. 
- In `token/smtp.json`, there should be the config of web maintainer's email sender. The format is: 
```json
{
  "host": "example.com",
  "port": 465,
  "username": "registration@example.com",
  "password": "anypassword"
}
```
- In `token/paypal.json`, there should be paypal sandbox's clinet ID and secret. In formal release, please replace `SandboxEnvironment` in `__init__` function in `paypal/models.py > PaypalClient` class with `LiveEnvironment`, and use live's clinet ID and secret in `token/paypal.json`. The format is:
```json
{
  "client_id": "anypassword",
  "secret": "anypassword"
}
```

If you don't use a registration confirming service by email, `smtp.json` is not necessary. However, you should delete `add_register`, `send_confirm_email` functions and `RegisterSheet` class, and modify `add_user` function to link the result of `LoginForm` directly.

### 5.2.	Build Python environment

Install required Python packages:

```
pip install -r requirements.txt
```

It is a maximum required package. With the environment, all functions can be used, but not all functions are necessary.

Navigate to the project folder, and create the database and super user:

```
python manage.py migrate
python manage.py createsuperuser
```

Follow the instructions in the command line. This user has the highest permission in this software.

### 5.3. Build static files

Replace `STATICFILES_DIRS = ['templates/static']` with `STATIC_ROOT = 'templates/static'` in `question_go/settings.py`.

Run the command: 
```
python manage.py collectstatic
```

Replace `STATIC_ROOT = 'templates/static'` with `STATICFILES_DIRS = ['templates/static']` in `question_go/settings.py`.

Replace `DEBUG = True` with `DEBUG = False` in `question_go/settings.py`.

### 5.4. Download pre-trained models

### 5.5. Administrator's settings

Run the command: 
```
python manage.py 0.0.0.0:$port --insecure
```
The IP address can only be 127.0.0.1 (for local use only) or 0.0.0.0 (for web server), and `port` can be customized.

Visit https://example.com:$port/admin. Create at least one group. Add the groups, which users can freely add into, to "Register groups" table. These groups each must include the following permissions:
- Library: add, delete, change, view papers
- My login: add, change, view register
- Paypal: view plans; add, delete, change, view subscription; add, change, view transaction
- Task manager: add, delete, change, view openedtask; add, delete, change, view step; add, delete, change, view task

Add charged functions to some new groups, and bound them to plans. Add all groups that contains charged functions to "Locked groups" table. Everytime when users login, the software will remove them from all locked groups, and then add them back according to effective subscriptions.

** question-go functional settings **

Set group storages for each group, which represents the storage each user in this group can use. 
