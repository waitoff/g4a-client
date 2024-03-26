#!D:\work\.virtualenvs\g4a-client\Scripts\python.exe
#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
__version__ = '0.0.2'
import os
import webbrowser
import secrets
import string

from threading import Timer
from flask_bootstrap import Bootstrap4 as Bootstrap
from flask_apscheduler import APScheduler
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_avatars import Avatars
from invoke import run
from web3 import Web3
# from web3.middleware import geth_poa_middleware
# from eth_account.messages import encode_defunct

load_dotenv(override=True)
port = int(os.environ.get("UVICORN_PORT", 15015))
host = os.environ.get("UVICORN_HOST", "127.0.0.1")
debug = bool(os.environ.get("UVICORN_DEBUG", False))
server_url = os.environ.get("SERVER_URL", "http://127.0.0.1:10101")
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
    print('Job 1 executed')


@app.route("/")
def home():
    global user_id
    # make User ID random if not exists
    env_file_path = '.env'
    if not os.path.exists(env_file_path):
        ww3 = Web3()
        w = ww3.eth.account.create()
        new_user_id = str(w.address)
        if user_id is None:
            user_id = new_user_id
        env_variables = {
            "USER_ID": new_user_id,
            "USER_KEY": str(w.key.hex()),
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
    env_file_path = os.path.join(os.path.dirname(__file__), 'worker-gpu', ".env")
    if os.path.exists(env_file_path):
        os.remove(env_file_path)
    env_variables = {
        "USER_ID": user_id,
        "SERVER_URL": server_url
    }
    with open(env_file_path, 'w') as f:
        for key, value in env_variables.items():
            f.write(f"{key}={value}\n")

    cmd = "call run.bat"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    print(result.stdout)
    return str(result.ok)


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

