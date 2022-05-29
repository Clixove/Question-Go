# Question Go

 Construct and reuse machine learning models

![](https://img.shields.io/badge/dependencies-Python%203.8--3.9-blue.svg)
![](https://img.shields.io/badge/dependencies-Django%203.2-green.svg)

Download the [documentation](https://github.com/Clixove/Question-Go/releases/download/v0.4.10/default.pdf).

## Citation

The programming language is Python 3.9, dependent packages are listed in `requirements.txt`. Besides:

Pandas profiling: https://github.com/pandas-profiling/pandas-profiling

## Usage

## Installation

The current folder of command line is the software's project root.

**Use Django token.**

Generate the following structure of files in project root, and include the following files in it.

```
token/
token/django_secret_key
token/smtp.json
```

*Notice the email registry module is outdated. For developers, please use "my_login" module in Question-Go repository and make proper modifications. For end users, please contact the author to obtain technical supprt.*

`django_secret_key` is a plain text file which contains a token, generated at [Djecrety](https://djecrety.ir/) or a random string of 52 characters. `smtp.json` is the configuration file of emal registry module, using the following format.

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

Create `token/paypal.json` and configure the password as the following format.

```json
{
"client_id": "anypassword",
"secret": "anypassword"
}
```

When in real business, replace `paypal/models.py -> PaypalClient class -> SandboxEnvironment` to `LiveEnvironment`. Also, use live environment's client ID and secret in `token/paypal.json`.

**Install Python environment.**

Install required Python packages.

```
pip install -r requirements.txt
```

*It is a maximum required package. With the environment, all functions can be used, but not all functions are necessary.*

Navigate to the project folder, and create the database and superuser.

```
python manage.py migrate
python manage.py createsuperuser
```

Follow the instructions in the command line to create the super administrator.

Start the website with the following command, where `$port` should be replaced with customized port number.

```
python manage.py 0.0.0.0:$port --insecure
```

Visit `https://example.com:$port/main` to preview the index page, and visit `https://example.com:$port/admin` to configure registry permission.

**Administrator's configuration.**

Create at least one user group. Basically, user permissions should include "add, change, view Register", "add, delete, change, view Task", "add, delete, change, view AsyncErrorMessage", and   "add, delete, change, view Column".

Create a register group, and link to proper user groups.

**Question-go functional settings**

Set group storages for each group, which represents the storage each user in this group can use. 
