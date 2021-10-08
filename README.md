# QuestionGo
 Construct and reuse machine learning models

![](https://img.shields.io/badge/dependencies-Python%203.8--3.9-blue)
![](https://img.shields.io/badge/dependencies-Django%203.2-green)
![](https://img.shields.io/badge/tests-Chrome%2089--92%20%E2%9C%94-brightgreen)

Document coming up soon ...

## Acknowledge

[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

## Install

The `token/` folder is hidden, because it includes passwords and keys. 
These files should be in this folder:
- `django_secret_key` It contains a string encrypting sessions and cookies, 
  and can be generated in [Djecrety](https://djecrety.ir/).
  
- `smtp.json` It contains the authentication method of the sender of registration confirming email. It should be like follows:
  
    ```json
  {
  "host": "smtp.example.com",
  "port": 465,
  "username": "sender@example.com",
  "password": "any"
  }
    ```
  
- `paypal.json` It contains the `client_id` and `secret` of the Paypal sandbox application created by business account. Referring to https://developer.paypal.com/home

  ```json
  {
  "client_id": "token",
  "secret": "token"
  }
  ```

  It uses sandbox environment in this repository. If using in live environment, replace tokens with a Paypal live application, and replace 

  ```python
  self.environment = SandboxEnvironment(client_id=self.client_id, client_secret=self.client_secret)
  ```

  with

  ```python
  self.environment = LiveEnvironment(client_id=self.client_id, client_secret=self.client_secret)
  ```

  It is in Line.64 (when defining class `PaypalClient`) of file `paypal/models.py` in v0.3.0.

`${...}` contains variables that you need to replace according to your 
environment.

```bash
cd ${project-base-folder}
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
