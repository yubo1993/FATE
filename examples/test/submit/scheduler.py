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
import collections
import heapq
import typing
from enum import Enum

from examples.test.submit.task import AbstractSubmitTask, AbstractSubmitResult
from examples.test.utils.async_processor import FailTaskResult


class Status(Enum):
    PENDING = 0
    CONDITION_WAITING = 1
    SUBMIT_READY = 2
    SUBMITTED = 3
    CANCELED = 4
    FAIL = 5
    SUCCESS = 6


class TaskScheduler(object):
    def __init__(self, tasks: typing.List[AbstractSubmitTask]):
        self.tasks: typing.Mapping[str, AbstractSubmitTask] = {task.task_id: task for task in tasks}
        self.result: typing.MutableMapping[str, AbstractSubmitResult] = {}
        self.status = {task.task_id: Status.PENDING for task in tasks}

        self.submit_ready: collections.deque() = []  # container for heapq
        self.waiting_taskid_set: typing.MutableSet[str] = set()
        self.post_task_map: typing.MutableMapping[str, typing.List[str]] = {}

        # find non-depending task
        for taskid, task in self.tasks.items():
            if len(task.pre_tasks) > 0:  # has depending
                self.status[taskid] = Status.CONDITION_WAITING
                for pre_taskid in task.pre_tasks:
                    if pre_taskid not in self.post_task_map:
                        self.post_task_map[pre_taskid] = []
                    self.post_task_map[pre_taskid].append(taskid)
                    self.waiting_taskid_set.add(taskid)
            else:
                self.status[taskid] = Status.SUBMIT_READY
                heapq.heappush(self.submit_ready, taskid)
        print("aa")

    def _dependence_satisfied(self, task_id):
        task = self.tasks[task_id]
        for pre_taskid in task.pre_tasks:
            if self.status[pre_taskid] != Status.SUCCESS:
                return False
        return True

    def update_task_submit_result(self, task_result: AbstractSubmitResult):
        self.result[task_result.task_id] = task_result
        if not isinstance(task_result, FailTaskResult) and task_result.status == "success":
            self.task_success(task_result.task_id)
        else:
            self.task_fail(task_result.task_id)

    def task_success(self, task_id):
        self.status[task_id] = Status.SUCCESS

        # update post-task status
        for post_taskid in self.post_task_map[task_id]:
            if post_taskid in self.waiting_taskid_set and self._dependence_satisfied(post_taskid):
                self.waiting_taskid_set.remove(post_taskid)
                heapq.heappush(self.submit_ready, post_taskid)
                self.status[post_taskid] = Status.SUBMIT_READY
                post_task = self.tasks[post_taskid]
                post_task.render_depends([self.result[task_id] for task_id in post_task.pre_tasks])

    def task_fail(self, task_id):
        self.status[task_id] = Status.FAIL
        canceled_taskid_set = set(self.post_task_map.get(task_id, []))
        while len(canceled_taskid_set) > 0:
            canceled_taskid = canceled_taskid_set.pop()
            self.status[canceled_taskid] = Status.CANCELED
            if canceled_taskid in self.waiting_taskid_set:
                self.waiting_taskid_set.remove(canceled_taskid)
                for _post_id in self.post_task_map.get(canceled_taskid, []):
                    if self.status[_post_id] == Status.CONDITION_WAITING:
                        canceled_taskid_set.add(_post_id)

    def has_ready(self):
        return len(self.submit_ready) > 0

    def has_waiting(self):
        return len(self.waiting_taskid_set) > 0

    def next_ready(self):
        taskid = heapq.heappop(self.submit_ready)
        self.status[taskid] = Status.SUBMITTED
        return self.tasks[taskid]
