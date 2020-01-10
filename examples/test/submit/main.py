#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import argparse
import os
import queue
import time
from atexit import register

from examples.test.submit.client import Submitter, Env
from examples.test.submit.parser import test_suite_parser
from examples.test.submit.scheduler import TaskScheduler, Status
from examples.test.submit.task import SubmitTaskHandler, AbstractSubmitResult
from examples.test.utils.async_processor import AsyncProcessor, FailTaskResult

TEST_SUITE_SUFFIX = "testsuite.json"


def search_test_suite(file_dir, suffix=TEST_SUITE_SUFFIX):
    test_suites = []
    for root, dirs, files in os.walk(file_dir):
        for file_name in files:
            if file_name.endswith(suffix):
                path = os.path.join(root, file_name)
                test_suites.append(path)
    return test_suites


def main():
    example_path = "examples/federatedml-1.x-examples"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("env_conf", type=str, help="file to read env config")
    arg_parser.add_argument("-n", "--num", type=int, help="parallel", default=1)
    arg_parser.add_argument("-o", "--output", type=str, help="file to save result, defaults to `test_result`",
                            default="test_result")
    arg_parser.add_argument("-e", "--error", type=str, help="file to save error")
    arg_parser.add_argument("-m", "--mode", type=int, help="work mode", choices=[0, 1])
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dir", type=str, help="dir to find testsuites")
    group.add_argument("-s", "--suite", type=str, help="testsuite to run")
    arg_parser.add_argument("-i", "--interval", type=int, help="check job status every i seconds, defaults to 1",
                            default=1)
    arg_parser.add_argument("--skip_data", help="skip data upload", action="store_true")
    args = arg_parser.parse_args()

    env = Env.from_config(args.env_conf)
    if args.mode:
        env.work_mode = args.mode
    test_suites_dir = os.path.join(env.fate_home, example_path) if args.dir is None else args.dir

    test_suites = [args.suite] if args.suite else search_test_suite(test_suites_dir)
    tasks = []
    priority = 1
    for suite in test_suites:
        suite_tasks = test_suite_parser(suite, env, priority)
        for task in suite_tasks:
            if hasattr(task, "render"):
                task.render(env)
        priority += len(suite_tasks)
        tasks.extend(suite_tasks)

    scheduler = TaskScheduler(tasks)
    submitter = Submitter(env)

    @register
    def _on_exit():
        with open(args.output, "a") as f:
            f.write("\n")
            f.write(f"{time.strftime('%Y-%m-%d %X')}\n")
            f.write(f"===================\n")
            for taskid, status in scheduler.status.items():
                if status in [Status.CONDITION_WAITING, Status.SUBMIT_READY, Status.PENDING, Status.CANCELED]:
                    f.write(f"[{status}]{taskid}\n")
                if taskid in scheduler.result:
                    task_result = scheduler.result[taskid]
                    if isinstance(task_result, FailTaskResult):
                        f.write(f"[{status}]{taskid}-0-submit_fail\n")
                    else:
                        f.write(f"[{status}]{taskid}-{task_result.job_id}-{task_result.status}\n")

    with AsyncProcessor() as processor:
        for i in range(args.num):
            processor.add_task_handler(SubmitTaskHandler(submitter))
        while scheduler.has_ready() or scheduler.has_waiting():
            while scheduler.has_ready():
                task = scheduler.next_ready()
                print(f"[error]{task}")
                processor.add_task(task)
            try:
                while True:
                    result: AbstractSubmitResult = processor.result_queue.get_nowait()
                    scheduler.update_task_submit_result(result)
            except queue.Empty:
                pass
            time.sleep(0.1)
        print(scheduler.status)


if __name__ == "__main__":
    main()
