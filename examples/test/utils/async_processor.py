import abc
import asyncio
import math
import queue
import threading
import time
import typing

from examples.test.utils.log import Logger


class AbstractTask(Logger):
    def __init__(self, task_id: str, priority: typing.Union[int, float] = 0, timeout=math.inf):
        self.task_id = task_id
        self.priority = priority
        self._timeout = timeout
        self._start_time = time.time()

    def time_reset(self):
        self._start_time = time.time()

    def time_elapse(self):
        return time.time() - self._start_time

    def __str__(self):
        return f"<{self.__class__.__name__}: id={id(self)}, priority={self.priority}>"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, AbstractTask):
            raise ValueError(f"can't compare task with type{other}")
        return self.priority == other.priority

    def __lt__(self, other):
        if not isinstance(other, AbstractTask):
            raise ValueError(f"can't compare task with type{other}")
        return self.priority < other.priority

    @property
    def timeout(self):
        return self._timeout


class AbstractTaskResult(Logger):
    def __init__(self, task_id, priority=0):
        self.task_id = task_id
        self.priority = priority

    def __eq__(self, other):
        if not isinstance(other, AbstractTaskResult):
            raise ValueError(f"can't compare task with type{other}")
        return self.priority == other.priority

    def __lt__(self, other):
        if not isinstance(other, AbstractTaskResult):
            raise ValueError(f"can't compare task with type{other}")
        return self.priority < other.priority

    def __str__(self):
        return f"<{self.__class__.__name__}:{id(self)}>"


class _PoisonPill(AbstractTask):
    def __init__(self):
        super().__init__(task_id="_poison_pill", priority=math.inf)

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance


class FailTaskResult(AbstractTaskResult):
    def __init__(self, task_id, priority):
        super().__init__(task_id=task_id, priority=priority)


class TaskHandler(Logger):
    def __init__(self):
        self.name = None
        # noinspection PyTypeChecker
        self.result_queue: queue.Queue = None

    def set_name(self, name):
        self.name = name

    @abc.abstractmethod
    async def do_task(self, task: AbstractTask) -> AbstractTaskResult:
        pass


class AsyncProcessor(Logger):

    def __init__(self):
        super().__init__()
        self.result_queue = queue.Queue()  # queue to gather results, thread safe
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._job_queue = None
        self.num_task_handler = 0
        self.even_thread = None

    def __enter__(self) -> 'AsyncProcessor':
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elegant_stop()

    def start(self):
        self.even_thread = threading.Thread(target=self._start, args=(self._loop,), daemon=True)
        self.even_thread.start()
        return self

    def _start(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._job_queue = asyncio.PriorityQueue()
        self._loop.run_forever()

    def _run_coroutine(self, coroutine) -> asyncio.futures.Future:
        self._assert_alive()
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def add_task_handler(self, task_handler: TaskHandler):
        self._run_coroutine(self._add_task_handler(task_handler))

    def add_task(self, task: AbstractTask):
        return asyncio.wait(self._run_coroutine(self._add_task(task)))

    def add_tasks(self, tasks):
        futures = [self._run_coroutine(self._add_task(task)) for task in tasks]
        return asyncio.wait(futures)

    def stop_workers(self):
        return self.add_tasks([_PoisonPill() for _ in range(self.num_task_handler)])

    def join_job_queue(self):
        return self._run_coroutine(self._join_job_queue()).result()

    def stop(self):
        self._assert_alive()
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _assert_alive(self):
        if not self._loop.is_running():
            raise EnvironmentError("loop not running")

    def elegant_stop(self):
        self.stop_workers()
        self.join_job_queue()
        self.stop()

    """async call"""

    async def _add_task_handler(self, task_handler: TaskHandler):
        self.num_task_handler += 1
        task_handler_name = f"task-handler-{self.num_task_handler}"
        task_handler.set_name(task_handler_name)
        self.debug(f"[{task_handler_name}]start")

        while True:
            task: AbstractTask = await self._job_queue.get()
            task.time_reset()
            # receive poison pill, stop task handler
            if isinstance(task, _PoisonPill):
                self.debug(f"[{task_handler_name}]receive poison pill, die")
                self._job_queue.task_done()
                return

            self.debug(f"[{task_handler_name}]start {task}")

            result = FailTaskResult(task.task_id, task.priority)
            # noinspection PyBroadException
            try:
                result = await task_handler.do_task(task)
            except Exception:
                self.exception(f"[{task_handler_name}] error")
            finally:
                self._job_queue.task_done()
                self.result_queue.put(result)
            self.debug(f"[{task_handler_name}]done {task}")

    async def _add_task(self, task):
        self.debug(f"[add task]{task}")
        self._job_queue.put_nowait(task)

    async def _join_job_queue(self):
        await self._job_queue.join()

    async def _stop(self):
        self._loop.stop()
