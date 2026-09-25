# proton-camoufox

Python-библиотека для работы с Proton Mail через Camoufox и Playwright. Поддерживает вход, регистрацию с ручным прохождением проверки, просмотр и поиск писем. Независимый проект, не связанный с Proton.

## Установка

Требуется Python 3.10 или новее. После установки пакета загрузите браузер Camoufox:

```powershell
py -m pip install proton-camoufox
py -m camoufox fetch
```

Для установки из исходного кода используйте `py -m pip install .`. Файл `requirements.txt` содержит зависимости проекта для установки из каталога исходников.

## Быстрый старт

Перед запуском задайте переменные окружения `PROTON_USERNAME` и `PROTON_PASSWORD`. Например, в PowerShell:

```powershell
$env:PROTON_USERNAME = "example@proton.me"
$env:PROTON_PASSWORD = "ваш-пароль"
```

Пример входа и чтения последних писем:

```python
import os

from proton_camoufox import ProtonMailClient

with ProtonMailClient(
    username=os.environ["PROTON_USERNAME"],
    password=os.environ["PROTON_PASSWORD"],
    headless=False,
) as client:
    client.login(allow_manual_verification=True)
    for message in client.list_messages(limit=10):
        print(message.subject)
```

## Примеры

Примеры находятся в папке `examples/` исходного проекта:

- `examples/login.py` — вход и вывод тем последних писем;
- `examples/register.py` — регистрация;
- `examples/proxy.py` — вход через прокси с необязательной авторизацией на прокси.

Логин и пароль передаются через переменные окружения `PROTON_USERNAME` и `PROTON_PASSWORD`. Пример запуска в PowerShell:

```powershell
$env:PROTON_USERNAME = "example@proton.me"
$env:PROTON_PASSWORD = "ваш-пароль"
py examples/login.py
```

Для регистрации используйте желаемый логин в `PROTON_USERNAME` и запустите `py examples/register.py`. Чтобы сохранить фразу восстановления, задайте путь через `PROTON_RECOVERY_FILE`. Для примера с прокси задайте `PROXY_SERVER` (например, `http://127.0.0.1:8080`); при необходимости также задайте `PROXY_USERNAME` и `PROXY_PASSWORD`, затем запустите `py examples/proxy.py`.

Интерфейс Proton может меняться и влиять на работу автоматизации.
