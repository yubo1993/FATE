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
import asyncio
import json
import math
import os
import tempfile
import typing
from datetime import timedelta

from examples.test.submit.client import Submitter, Env
from examples.test.utils.async_processor import AbstractTask, AbstractTaskResult, TaskHandler


class JsonDumpedFile(object):
    def __init__(self, d: dict):
        self._f = tempfile.NamedTemporaryFile("w")
        self._d = d

    def __enter__(self):
        f = self._f.__enter__()
        json.dump(self._d, f)
        f.flush()
        return f

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._f.__exit__(exc_type, exc_val, exc_tb)


class SubmitTaskHandler(TaskHandler):

    def __init__(self, submitter: Submitter):
        super().__init__()
        self.submitter = submitter

    async def do_task(self, task: 'AbstractSubmitTask'):
        return await task.do_task(self.submitter)


class AbstractSubmitTask(AbstractTask):
    def __init__(self, task_id, pre_tasks=None, priority: typing.Union[int, float] = 0, timeout=math.inf):
        super().__init__(task_id=task_id, priority=priority, timeout=timeout)
        self.pre_tasks = [] if pre_tasks is None else pre_tasks

    def __str__(self):
        return f"<{self.__class__.__name__}: id={id(self)}, pre_tasks={self.pre_tasks}, priority={self.priority}>"

    async def do_task(self, submitter: Submitter) -> 'AbstractSubmitResult':
        raise NotImplementedError()

    async def check_job_status(self, job_id, submitter: Submitter):
        while True:
            stdout = submitter.submit(["-f", "query_job", "-j", job_id])
            status = stdout["data"][0]["f_status"]
            elapse_seconds = self.time_elapse()
            self.info(f"{job_id} {status}, elapse: {timedelta(seconds=elapse_seconds)}")
            if (status == "running" or status == "waiting") and elapse_seconds < self.timeout:
                await asyncio.sleep(0.1)
            else:
                return status

    def render_depends(self, depend_results: typing.List['AbstractSubmitResult']):
        pass


class AbstractSubmitResult(AbstractTaskResult):
    def __init__(self, task_id, status, job_id, priority=0):
        super().__init__(task_id=task_id, priority=priority)
        self.status = status
        self.job_id = job_id


class LocalDataUploadTask(AbstractSubmitTask):

    def __init__(self, task_id, conf: dict, priority=0, timeout=math.inf):
        super().__init__(task_id=task_id, priority=priority, timeout=timeout)
        self.conf = conf

    def render(self, env: Env):
        self.conf["work_mode"] = env.work_mode
        self.conf["backend"] = env.backend
        return self

    def __str__(self):
        return f"{super().__str__()}: conf={self.conf}"

    async def do_task(self, submitter: Submitter):
        with JsonDumpedFile(self.conf) as c:
            stdout = submitter.submit(["-f", "upload", "-c", c.name])
        job_id = stdout['jobId']
        status = await self.check_job_status(job_id, submitter)
        return DataUploadResult(task_id=self.task_id, status=status, job_id=job_id)


class RemoteDataUploadTask(AbstractSubmitTask):

    def __init__(self, task_id, remote_host, conf, priority=0, timeout=math.inf):
        super().__init__(task_id=task_id, priority=priority, timeout=timeout)
        self.remote_host = remote_host
        self.conf = conf

    async def do_task(self, submitter: Submitter):  # todo: enhance
        with JsonDumpedFile(self.conf) as c:
            submitter.run_cmd(["scp", c.name, f"{self.remote_host}:{c.name}"])
            env_path = os.path.join(submitter.fate_home, "../../init_env.sh")
            upload_cmd = " && ".join([f"source {env_path}",
                                      f"python {submitter.flow_client_path} -f upload -c {c.name}",
                                      f"rm {c.name}"])
            stdout = submitter.run_cmd(["ssh", self.remote_host, upload_cmd])
            try:
                stdout = json.loads(stdout)
                status = stdout["retcode"]
            except json.decoder.JSONDecodeError:
                raise ValueError(f"[submit_job]fail, stdout:{stdout}")
            if status != 0:
                raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
            job_id = stdout['jobId']
            status = await self.check_job_status(job_id, submitter)
        return DataUploadResult(task_id=self.task_id, status=status, job_id=job_id)


class DataUploadResult(AbstractSubmitResult):
    def __str__(self):
        return f"{self.__class__.__name__}(job_id={self.job_id}, status={self.status})"


class TrainTask(AbstractSubmitTask):

    def __init__(self, task_id, conf, dsl, pre_tasks=None, priority=0, timeout=math.inf):
        super().__init__(task_id=task_id, pre_tasks=pre_tasks, priority=priority, timeout=timeout)
        self.conf = conf
        self.dsl = dsl

    def __str__(self):
        return f"{super().__str__()}: conf={self.conf}, dsl={self.dsl}"

    async def do_task(self, submitter: Submitter):
        with JsonDumpedFile(self.conf) as c:
            with JsonDumpedFile(self.dsl) as d:
                stdout = submitter.submit(["-f", "submit_job", "-c", c.name, "-d", d.name])
        model_info = stdout["data"]["model_info"]
        job_id = stdout['jobId']
        status = await self.check_job_status(job_id, submitter)
        return TrainResult(task_id=self.task_id, status=status, model_info=model_info, job_id=job_id)

    def render(self, env):
        self.conf["job_parameters"]["work_mode"] = env.work_mode
        self.conf["job_parameters"]["backend"] = env.backend
        self.conf["initiator"]["role"] = "guest"
        self.conf["initiator"]["party_id"] = env.role["guest"][0]

        for role in ["guest", "host", "arbiter"]:
            if role in self.conf["role"]:
                num = len(self.conf["role"][role])
                self.conf["role"][role] = env.role[role][:num]


class TrainResult(AbstractSubmitResult):
    def __init__(self, task_id, status, job_id, model_info=None, priority=0):
        super().__init__(task_id=task_id, status=status, job_id=job_id, priority=priority)
        self.model_info = model_info


class PredictTask(AbstractSubmitTask):

    def __init__(self,
                 task_id: str,
                 conf: dict,
                 pre_tasks: typing.List[str],
                 priority: int = 0,
                 timeout: float = math.inf):
        super().__init__(task_id=task_id, pre_tasks=pre_tasks, priority=priority, timeout=timeout)
        self.conf = conf

    def __str__(self):
        return f"{super().__str__()}: conf={self.conf}"

    async def do_task(self, submitter: Submitter):
        with JsonDumpedFile(self.conf) as c:
            stdout = submitter.submit(["-f", "submit_job", "-c", c.name])
        job_id = stdout['jobId']
        status = await self.check_job_status(job_id, submitter)
        return PredictResult(task_id=self.task_id, status=status, job_id=job_id)

    def render(self, env):
        self.conf["job_parameters"]["work_mode"] = env.work_mode
        self.conf["job_parameters"]["backend"] = env.backend
        self.conf["job_parameters"]["job_type"] = "predict"
        self.conf["initiator"]["role"] = "guest"
        self.conf["initiator"]["party_id"] = env.role["guest"][0]

        for role in ["guest", "host", "arbiter"]:
            if role in self.conf["role"]:
                num = len(self.conf["role"][role])
                self.conf["role"][role] = env.role[role][:num]

    def render_depends(self, depend_results: typing.List['AbstractSubmitResult']):
        for depend_result in depend_results:
            if isinstance(depend_result, TrainResult):
                self.set_model_info(depend_result.model_info)
                break
        else:
            raise ValueError("non train results found")

    def set_model_info(self, model_info):
        self.conf["job_parameters"]["model_id"] = model_info["model_id"]
        self.conf["job_parameters"]["model_version"] = model_info["model_version"]


class PredictResult(AbstractSubmitResult):
    pass
