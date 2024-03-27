#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
__version__ = '0.0.3'
import os
import sys
import webbrowser
import secrets
import string
import requests
import subprocess
#
from threading import Timer
from flask_bootstrap import Bootstrap4 as Bootstrap
from flask_apscheduler import APScheduler
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_avatars import Avatars
from invoke import run
# local imports
# import worker

load_dotenv(override=True)
port = int(os.environ.get("UVICORN_PORT", 15015))
host = os.environ.get("UVICORN_HOST", "127.0.0.1")
debug = bool(os.environ.get("UVICORN_DEBUG", False))
server_url = os.environ.get("SERVER_URL", "https://manager.g4a.auk.su")
user_id = os.environ.get("USER_ID")
# SECRET_KEY = 'L7qi1RC7b6B24UFfBUu69497e72H2NFhbUlO6P9GkP86isf7i2'
SECRET_KEY = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))

app = Flask(__name__, static_folder="static")
app.config.from_object(__name__)
bootstrap = Bootstrap(app)
avatars = Avatars(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


@scheduler.task('interval', id='do_job_1', seconds=30, misfire_grace_time=900)
def job1():
    # print('Job 1 executed')
    pass


@app.route("/")
def home():
    global user_id
    rnd_id = '6546546546546546541'
    # work_path = os.path.dirname(os.path.abspath(__file__))
    # print(work_path)
    env_file_path = '.env'
    if not os.path.exists(env_file_path):
        r = requests.get(server_url + '/register/' + rnd_id).text
        new_user_id = r.split(':')[0]
        new_key = r.split(':')[1]
        env_variables = {
            "USER_ID": new_user_id,
            "USER_KEY": new_key,
            "SERVER_URL": server_url
        }
        with open('.env', 'w') as f:
            for key, value in env_variables.items():
                f.write(f"{key}={value}\n")

    return render_template('index.html')


@app.route("/profile")
def get_profile():
    profile = {
        'name': 'None',
        'id': user_id
    }
    return render_template('profile.html', profile=profile)


@app.route("/start")
def start():
    cmd = "python worker.py"
    result = run(cmd, hide=False, warn=True)
    # cmd = "call run.bat"
    # result = run(cmd, hide=False, warn=True)
    # print(result.stdout)
    print(result.ok)
    return 'ok'


@app.route("/update")
def update():
    print('Update')
    cmd = "update.bat"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    print(result.stdout)
    return str(result.ok)


def open_browser():
    webbrowser.open_new(f'http://{host}:{port}')


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=debug, port=port, host=host)

