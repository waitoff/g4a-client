#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
__version__ = '0.1.4'
import os
import httpx
import webbrowser
import secrets
import string
#
from threading import Timer
from flask_bootstrap import Bootstrap4 as Bootstrap
from flask_apscheduler import APScheduler
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_avatars import Avatars
from invoke import run
from web3 import Web3
# local imports

load_dotenv(override=True)
port = int(os.environ.get("UVICORN_PORT", 15015))
host = os.environ.get("UVICORN_HOST", "127.0.0.1")
debug = bool(os.environ.get("UVICORN_DEBUG", False))
server_url = os.environ.get("SERVER_URL", "https://manager.g4a.auk.su")
user_id = os.environ.get("USER_ID")
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
    """
    A task decorator that schedules a job to be executed at fixed intervals.

    Parameters:
        id (str): The unique identifier for the job.
        seconds (int): The interval in seconds between each execution of the job.
        misfire_grace_time (int): The grace period in seconds before a missed execution is considered a misfire.

    Returns:
        None
    """
    # print('Job 1 executed')
    pass


@app.route("/")
def home():
    """
    Renders the home page of the application.

    This function is the route handler for the root URL ("/"). It checks if the environment file exists, and if not, it creates a new user ID and key using the Web3 library. If the global variable `user_id` is None, it assigns the newly created user ID to it. It then creates a dictionary of environment variables with the user ID, user key, and server URL. Finally, it writes these environment variables to the .env file and renders the 'index.html' template.

    Returns:
        The rendered 'index.html' template.

    """
    global user_id
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
    """
    Retrieves the user profile information and renders it in the 'profile.html' template.

    Returns:
        The rendered 'profile.html' template with the user profile information.
    """
    profile = {
        'name': 'None',
        'id': user_id
    }
    return render_template('profile.html', profile=profile)


@app.route("/start")
def start():
    """
    A function that handles the "/start" route.

    This function is responsible for launching the worker process by executing the "worker.py" script. It uses the `run` function from the `subprocess` module to run the command "python worker.py" and captures the output. The function then prints the result of the command execution and returns the string "Worker started".

    Returns:
        str: The string "Worker started".
    """
    print('Launch worker')
    cmd = "python worker.py"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    return 'Worker started'


@app.route("/update")
def update():
    """
    Route decorator for the "/update" endpoint.

    This function is responsible for handling the "/update" route. It executes the "update.bat"
    command and captures the output. The function then prints the result of the command execution
    and returns a string representation of the command's success status.

    Returns:
        str: A string representation of the command's success status.
    """
    print('Update')
    cmd = "..\\update.bat"
    result = run(cmd, hide=False, warn=True)
    print(result.ok)
    print(result.stdout)
    return str(result.ok)


@app.route("/stat")
def stat():
    """
    A function to make a GET request to the server for statistics data and display it in an HTML container.
    """
    r = httpx.get(server_url + '/users/stat', headers={'authorization': user_id})
    return f'''
    <div class="container">
        <h4>Stats:</h4>
        <p>Number of workers: <span id="workers_count">{r.json()['workers_count']}</span></p>
        <p>Number of works: <span id="works_count">{r.json()['works_count']}</span></p>
        <p>Number of solutions: <span id="solution_count">{r.json()['solution_count']}</span></p>
        <p>Average work time: <span id="avg_work_time">{r.json()['avg_work_time']}</span></p>
    </div>
    '''


def open_browser():
    """
    Opens a new browser window with the specified host and port.

    This function uses the `webbrowser.open_new` method to open a new browser window
    with the URL composed of the `host` and `port` variables. The `host` and `port`
    variables are used to construct the URL in the format `http://{host}:{port}`.

    Parameters:
        None

    Returns:
        None
    """
    webbrowser.open_new(f'http://{host}:{port}')


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=debug, port=port, host=host)
