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
import subprocess

from examples.test.utils.log import Logger

DEFAULT_FATE_HOME = os.path.abspath(os.path.join(os.getcwd(), "../../.."))


class Env(object):
    def __init__(self, role, ip_map=None, fate_home=DEFAULT_FATE_HOME, work_mode=0, backend=0):
        self.fate_home = fate_home
        self.work_mode = work_mode
        self.backend = backend
        self.role = role
        self.ip_map = {} if ip_map is None else ip_map
        for _, pid_list in role.items():
            for pid in pid_list:
                pid = str(pid)
                if pid not in self.ip_map:
                    self.ip_map[pid] = -1

    @classmethod
    def from_config(cls, config_path="../default_env.json"):
        with open(config_path) as f:
            config_dict = json.load(f)
        fate_home = config_dict.get("fate_home", DEFAULT_FATE_HOME)
        work_mode = config_dict.get("work_mode", 0)
        backend = config_dict.get("backend", 0)
        role = config_dict["role"]
        ip_map = config_dict.get("ip_map", None)
        return Env(role, ip_map, fate_home, work_mode, backend)

    def get_party_host(self, role=None, idx=None, party_id=None):
        if role is not None:
            if idx is not None:
                if role not in self.role:
                    raise ValueError(f"{role} not found in env.role")
                if idx >= len(self.role[role]):
                    raise ValueError(f"{idx} out of bound")
                party_id = str(self.role[role][idx])
        if party_id not in self.ip_map:
            raise ValueError(f"{party_id} not found in env.ip_map")
        return self.ip_map[party_id]


class Submitter(Logger):

    def __init__(self, env: Env):
        self.env = env

    @property
    def fate_home(self):
        return self.env.fate_home

    @property
    def flow_client_path(self):
        return os.path.abspath(os.path.join(self.fate_home, "fate_flow/fate_flow_client.py"))

    @staticmethod
    def run_cmd(cmd):
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        return stdout.decode("utf-8")

    def submit(self, cmd):
        full_cmd = ["python", self.flow_client_path]
        full_cmd.extend(cmd)
        self.debug(f"submit {full_cmd}")
        stdout = self.run_cmd(full_cmd)
        try:
            stdout = json.loads(stdout)
            status = stdout["retcode"]
        except json.decoder.JSONDecodeError:
            raise ValueError(f"[submit_job]fail, stdout:{stdout}")
        if status != 0:
            raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
        return stdout
