# #!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
__version__ = "1.1.4.GPU"

import base64
import sys
import os
import httpx
import random
#
from rich import print
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
from cryptography.fernet import Fernet
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from aiohttp import ClientSession
# local import
import ai

try:
    import asyncio
except ImportError:
    import trollius as asyncio

from dotenv import load_dotenv
load_dotenv()

my_id = os.getenv("USER_ID", "DEMO")
server_url = os.getenv("SERVER_URL", "http://127.0.0.1:10101")
work_limit = os.getenv("WORK_LIMIT", 1)
task_period = os.getenv("WORK_PERIOD", 2)
image_dir = os.environ.get("UPLOADER_IMAGE_DIR", "img")
fernet_key = os.environ.get("FERNET_KEY", b'7Ql80xScamifSl7Q_Q45PgF2zjfI2GdZZ2wuqo0mPjo=')

work_code = ''
my_config = {}
get_stat_period = 180


def get_id():
    """
    A function to get system information like memory, CPU, GPU, and OS details.
    """
    import psutil
    import platform
    import GPUtil

    global my_config
    global my_id
    global work_code

    uname = platform.uname()
    mem = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    gpu_count = len(gpus)
    if gpu_count > 0:
        gpu_type = gpus[0].name
        min_device = 'GPU'
        gpu_memory = gpus[0].memoryTotal
        serial = gpus[0].uuid
    else:
        gpu_type = 'None'
        min_device = 'CPU'
        gpu_memory = 0
        serial = 'None'

    my_config = {
        "min_memory": int(round(mem.total/1024/1024/1024, 0)),
        "cpu_count": psutil.cpu_count(logical=True),
        "min_device": min_device,
        "min_device_memory": int(round(gpu_memory/1024, 0)),
        "os": uname.system,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "gpu_serial": serial
    }
    get_stat()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))


def get_stat():
    """
    A function to retrieve statistics using an API call with given parameters and display the results.
    """
    global my_id
    try:
        with httpx.Client() as client:
            config = str(my_config)
            request = {'jsonrpc': '2.0', 'method': 'stat', 'params': [my_id, config], 'id': 1}
            s = client.post(url=server_url + "/api/rpc", json=request)
            response = s.json()
            id_panel = Panel(Text(f"My ID: {my_id} \n"
                                  f"My status             : {response['result']['status']}\n"
                                  f"- My tasks            : {response['result']['work_count']}\n"
                                  f"- My solutions        : {response['result']['answer_count']}\n"
                                  f"Script version        : {__version__}\n"
                                  f"Work Server           : {server_url}\n"
                                  f"My Config              \n"
                                  f"- os                  : {my_config['os']}\n"
                                  f"- memory              : {my_config['min_memory']}\n"
                                  f"- cpu_count           : {my_config['cpu_count']}\n"
                                  f"- device              : {my_config['min_device']}\n"
                                  f"- gpu_count           : {my_config['gpu_count']}\n"
                                  f"- gpu_memory          : {my_config['min_device_memory']}\n"
                                  f"- gpu_type            : {my_config['gpu_type']}\n"
                                  f"- gpu_serial          : {my_config['gpu_serial']}\n"
                                  f"",
                                  justify="left"))
            print(id_panel)

    except Exception as e:
        print(e)

    if my_id == '':
        print("Identifier not found")
        sys.exit(1)
    return


async def get_task():
    """
    A function to get a task from the server, process it, and complete it.
    This function interacts with the server using async requests and handles various responses.
    Returns True upon successful completion.
    """
    async def encode_file_base64(file_path):
        with open(file_path, "rb") as file:
            encoded_bytes = base64.b64encode(file.read())
            encoded_string = encoded_bytes.decode("utf-8")
            return encoded_string

    global my_id
    global work_code
    if my_id == '':
        return

    try:
        async with ClientSession() as session:
            # print('-------------- 1 get_work ---------------')
            config = my_config
            request = {'jsonrpc': '2.0', 'method': 'get_work', 'params': [my_id, config], 'id': 1}
            # result_file_name = ''
            # result_file_size = 0
            # encoded_file = ''

            async with session.post(url=server_url + "/api/rpc", json=request) as resp:
                response = await resp.json()
                # print('response=', response['result'])
                if not response['result']:
                    # print('No tasks found')
                    return
                else:
                    # print(response['result'])
                    # print(type(response['result']['parameters']))
                    print(response['result']['algorithm'])
                    work_guid = response['result']['id']
                    # print(response['result'])
                    # result = None
                    result = await process_task(
                        algorithm=response['result']['algorithm'],
                        parameters=response['result']['parameters']
                    )
                    print('process_task_result', str(result))
                    result_file_name = result[0]
                    encoded_file = await encode_file_base64(file_path=os.path.join(image_dir, result_file_name))
                    result_file_size = len(encoded_file)

            # print('-------------- 4 complete ---------------')
            complete = {
                'work_guid': work_guid,
                'result': result_file_name,
                'file': encoded_file,
                'file_size': result_file_size,
                'file_name': result_file_name,
                'file_type': '.png',
                'file_crc': '0x0',
                'file_url': ''
            }
            request = {'jsonrpc': '2.0', 'method': 'complete', 'params': [my_id, complete], 'id': 1}
            async with session.post(server_url + '/api/rpc', json=request) as c:
                response = await c.json()
                print('complete_id=', response['result'], ', result=', complete['result'])

    except Exception as e:
        print(e)

    return True


async def estimate_pi(n):
    num_point_circle = 0  # кол-во точек внутри круга
    num_point_total = n

    for i in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x ** 2 + y ** 2

        if distance <= 1:  # определяем, что точка попала внутрь круга
            num_point_circle += 1

    return 4 * num_point_circle / num_point_total


async def process_task(algorithm, parameters: str):
    """
    Process a task based on the specified algorithm and parameters.

    Args:
        algorithm (str): The algorithm to be used for processing the task.
        parameters (str): The parameters needed for the task.

    Returns:
        The result of processing the task based on the algorithm.
    """
    p = eval(parameters)
    if algorithm == 'PI':
        # n = 100000000  # общее количество точек (чем больше точек, тем лучше точность)
        n = 10000000
        return await estimate_pi(n)

    if algorithm == 'QR':
        return await ai.text_to_qr_local(
            prompt=p['prompt'],
            negative_prompt=p['negative_prompt'],
            image_file_name=p['image_file_name'],
            qr_code_text=p['qr_code_text'],
            controlnet=float(p['effect']),
        )

if __name__ == '__main__':
    if not os.path.exists('.env'):
        print("No .env file found. Make sure (or copy from .env.example) to run 'client'. Exiting.")
        sys.exit(1)
    else:
        print("Starting...\n")

    # scheduler = BlockingScheduler()
    scheduler = AsyncIOScheduler(
        # job_defaults={'coalesce': False, 'max_instances': 3},
        executors={'processpool': ProcessPoolExecutor(max_workers=1)},
    )
    # scheduler = BackgroundScheduler()
    get_id()
    scheduler.add_job(get_stat, 'interval', seconds=int(get_stat_period))
    scheduler.add_job(get_task, 'interval', seconds=int(task_period))
    scheduler.start()

    try:
        asyncio.get_event_loop().run_forever()
        # # This is here to simulate application activity (which keeps the main thread alive).
        # while True:
        #     time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
