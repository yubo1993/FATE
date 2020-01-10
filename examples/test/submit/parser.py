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
import json
import os

from examples.test.submit.task import LocalDataUploadTask, RemoteDataUploadTask, PredictTask, TrainTask


def test_suite_parser(suite, env, init_priority):
    priority = init_priority
    tasks = []
    test_suite_base_path = os.path.dirname(suite)
    with open(suite) as f:
        configs = json.loads(f.read())

    data_task = []
    if "data" in configs:
        for idx, data_config in enumerate(configs['data']):
            if 'role' not in data_config:
                raise ValueError(f"[{suite}][{configs}]role not found")
            role, role_id = data_config['role'].split("_", 1)
            host = env.get_party_host(role, int(role_id))
            del data_config['role']
            task_id = f"data-{idx}@{suite}"
            data_task.append(task_id)
            if host == -1:
                tasks.append(
                    LocalDataUploadTask(task_id=task_id, conf=data_config, priority=priority))
            else:
                tasks.append(
                    RemoteDataUploadTask(task_id=task_id, remote_host=host, conf=data_config, priority=priority))
            priority += 1

    if "tasks" in configs:
        for task_name, task_config in configs['tasks'].items():
            task_id = f"{task_name}@{suite}"
            if "conf" not in task_config:
                raise ValueError(f"[{suite}][{task_config}]conf not found")
            with open(os.path.join(test_suite_base_path, task_config["conf"])) as c:
                conf = json.load(c)

            if "deps" in task_config:
                pre_task = [f"{task_config['deps']}@{suite}"]
                pre_task.extend(data_task)
                tasks.append(PredictTask(task_id=task_id, conf=conf, pre_tasks=pre_task, priority=priority))

            else:
                if "dsl" not in task_config:
                    raise ValueError(f"[{suite}][{task_config}]dsl not found")
                with open(os.path.join(test_suite_base_path, task_config["dsl"])) as d:
                    dsl = json.load(d)
                tasks.append(TrainTask(task_id=task_id, conf=conf, pre_tasks=data_task, dsl=dsl, priority=priority))
            priority += 1

    return tasks
