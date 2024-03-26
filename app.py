#!D:\work\.virtualenvs\g4a-client\Scripts\python.exe
#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
__version__ = '0.0.1'
import os
import webbrowser

from threading import Timer
from flask_bootstrap import Bootstrap4 as Bootstrap
from flask_apscheduler import APScheduler
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_avatars import Avatars
from invoke import run


load_dotenv(override=True)
port = int(os.environ.get("UVICORN_PORT", 15015))
host = os.environ.get("UVICORN_HOST", "127.0.0.1")
debug = bool(os.environ.get("UVICORN_DEBUG", False))
SECRET_KEY = 'L7qi1RC7b6B24UFfBUu69497e72H2NFhbUlO6P9GkP86isf7i2'

app = Flask(__name__, static_folder="static")
app.config.from_object(__name__)
bootstrap = Bootstrap(app)
avatars = Avatars(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


@scheduler.task('interval', id='do_job_1', seconds=30, misfire_grace_time=900)
def job1():
    print('Job 1 executed')


# upload_folder = os.path.join('static', 'images')
# os.makedirs(upload_folder, exist_ok=True)
@app.route("/")
def home():
    return render_template('index.html')


@app.route("/profile")
def get_profile():
    profile = {
        'name': 'John Doe',
        'wallet_address': '0x1234567890123456789012345678901234567890',
        'email': 'nHb9w@example.com',
        'phone': '555-555-5555',
        'id': '1234567890'
    }
    return render_template('profile.html', profile=profile)


@app.route("/install")
def install():
    env_file_path = os.path.join(os.path.dirname(__file__), 'worker-gpu', ".env")
    if os.path.exists(env_file_path):
        os.remove(env_file_path)

    env_variables = {
        "USER_ID": "DEMO",
        "SERVER_URL": "http://127.0.0.1:10101"
    }
    with open(env_file_path, 'w') as f:
        for key, value in env_variables.items():
            f.write(f"{key}={value}\n")

    cmd = "call run.bat"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    print(result.stdout)
    return 'True'


@app.route("/example")
def example():
    print('Update')
    cmd = "update.bat"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    print(result.stdout)
    return 'True'
    # return result.stdout.decode('cp1251').encode('utf8').replace('\r\n', '<br>')
    # return '<button type="submit" onclick="myFunction()">Click me</button>'


def open_browser():
    webbrowser.open_new(f'http://{host}:{port}')


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=debug, port=port, host=host)

