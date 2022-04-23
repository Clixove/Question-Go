# Question

 Construct and reuse machine learning models

![](https://img.shields.io/badge/dependencies-Python%203.8--3.9-blue.svg)
![](https://img.shields.io/badge/dependencies-Django%203.2-green.svg)

Download the [documentation](https://github.com/Clixove/Question-Go/releases/download/v0.4.10/default.pdf).

## Citation

The programming language is Python 3.9, dependent packages are listed in `requirements.txt`. Besides:

Pandas profiling: https://github.com/pandas-profiling/pandas-profiling

## Usage

The current folder of command line is the software's project root.

**Build token files.**

1. Create `token` folder at project root.

2. Create `token/django_secret_key` and input 52 random characters.

3. Create `token/smtp.json` and configure the web maintainer's email sender as the following format.

```json
{
"host": "example.com",
"port": 465,
"username": "registration@example.com",
"password": "anypassword"
}
```

**Payment with PayPal.**

*Applicable only when the module `paypal`, `paypalhttp`, and `paypalcheckoutsdk` exist.*

1. Create `token/paypal.json` and configure the password as the following format.

```json
{
"client_id": "anypassword",
"secret": "anypassword"
}
```

2. If used in real business, replace `paypal/models.py -> PaypalClient class -> SandboxEnvironment` to `LiveEnvironment`. Also, use live environment's client ID and secret in `token/paypal.json`.

**Build the environment.**

```
pip install -r requirements.txt
```

*It is a complete environment. With the environment, all functions can be used, but not all functions are necessary.*

**Create the database.**

Navigate to the project folder, and create the database and super user:

```
python manage.py migrate
python manage.py createsuperuser
```

Follow the instructions in the command line. This user has the highest permission in this software.

### Administrator's settings

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

**Question-go functional settings**

Set group storages for each group, which represents the storage each user in this group can use. 
