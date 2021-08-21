# Question Go
 Construct and Reuse Machine Learning Models as a Workflow

![](https://img.shields.io/badge/dependencies-Python%203.8--3.9-blue)
![](https://img.shields.io/badge/dependencies-Django%203.2-green)
![](https://img.shields.io/badge/tests-Chrome%2089--92%20%E2%9C%94-brightgreen)

## Acknowledge

[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

## Install

The `token/` folder is hidden, because it includes passwords and keys. 
These files should be in this folder:
- `django_secret_key` It contains a string encrypting sessions and cookies, 
  and can be generated in [Djecrety](https://djecrety.ir/).
- `smtp.json` Write your SMTP config like the following. This email address 
  belongs to the website maintainer, and is used to send registration 
  confirming email to users and donation reminder to anyone (receivers in 
 `django database -> payment_donation -> Website Manager` table).
    ```json
    {
  "host": "smtp.example.com",
  "port": 465,
  "username": "sender@example.com",
  "password": "any"
    }
    ```
- `payment_methods/` This folder has been added to static file list. Write 
  your bank account (image, QR code) in blank HTML page, then save 
  the page into this folder. In `djngo database -> payment_donation -> Paying
  Method` table, `/static/` address is equal to path of this folder. If using
  direct link like `paypal.me`, just write the link in above table. 

`${...}` contains variables that you need to replace according to your 
environment.

```bash
cd ${project_base_folder}
pip install -r requirements.txt  # Note [1]
python manage.py migrate
python manage.py createsuperuser
# follow the instructions in command lines
python manage.py runserver
```

**Notes**:

[1] The provided `requirements.txt` is generated from a developing Anaconda.
It is not a minimal requirement because I remove some functions during 
programming, and the dependent packages cannot be removed clearly.

[2] The subscription module is available only for companies in China, mainland,
and have an account of [Alipay](https://b.alipay.com/index2.htm) (merchant 
version). If you enable this module, pay attention to `pycryptodome` and 
`pycryptodomex` packages (recommend fixing their versions).

## Application permissions

## Programming Instruction

### 1. Pre-processing workflow

To contribute us a data pre-processing workflow, please submit a minimal 
reproductive of your work in a `.py` file following these criteria:

```python
import pandas as pd
from django import forms

class DropColumns(PublicPreProcessing):
    pass


def pandas_function_name(config: forms.Form, dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [x.name for x in config.cleaned_data['targeted_columns']]
    # ------ Write Pre-processing Workflow START ------
    
    # ------ Write Pre-processing Workflow END   ------
    return dataframe
```

The Django form inherits from `PublicPreProcessing`. The variable names 
`targeted_column` and `algorithm` are occupied.
[Reference](https://github.com/Clixove/Question-Go/blob/fca897dd0b4107a41a71151c4086205a520ac422/pre_cross_sectional/views.py#L258)

To use customized parameters, replace `pass` with a statement of a parameter's
name and data type. The avaiable form fields are listed at: 
[Reference](https://docs.djangoproject.com/en/3.2/ref/forms/fields/) 

The `columns` variable is an example of obtaining parameters from the config 
form. When submitting your contributions, you can comment codes related to 
Django (used in online versions), and write a plain `pandas`

Finally, don't forget to give a name to your workflow. It's used in a 
dictionary  containing all installed workflows, an HTML template, and 
a corresponding web link. Therefore, it should be unique among all 
contributions.
