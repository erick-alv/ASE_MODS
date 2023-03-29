
import copy
class ImitPoseState:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._buffer = []
        self.start_pose = None

    # This function check if start poses has been saved;
    # this pose is used to initialize the body in the simulation
    def has_start_pose(self):
        return self.start_pose is not None

    # This function checks if the buffer has enough elements
    # so that one can pass them to the Policy
    def is_ready(self):
        return len(self._buffer) == self.num_timesteps

    def insert(self, element, transform_func=None, start_check_func=None):
        if transform_func is not None:
            element = transform_func(element)
        if self.start_pose is None and start_check_func is not None:
            if(start_check_func(element)):
                self.start_pose = element
        else:
            self._buffer.append(element)
            if len(self._buffer) > self.num_timesteps:
                self._buffer.pop(0)

    def get_start_pose(self):
        return copy.deepcopy(self.start_pose)

    def get(self):
        return self._buffer.copy()
        
        
import asyncio

class ImitPoseStateAsyncSafe(ImitPoseState):
    def __init__(self, num_timesteps):
        super().__init__(num_timesteps)
        self._lock = asyncio.Lock()

    async def has_start_pose(self):
        async with self._lock:
            return super().has_start_pose()

    async def is_ready(self):
        async with self._lock:
            return super().is_ready()

    async def insert(self, element, transform_func=None, start_check_func=None):
        async with self._lock:
            super().insert(element, transform_func, start_check_func)

    async def get_start_pose(self):
        async with self._lock:
            return super().get_start_pose()

    async def get(self):
        async with self._lock:
            return super().get()


import threading

class ImitPoseStateThreadSafe(ImitPoseState):
    def __init__(self, num_timesteps):
        super().__init__(num_timesteps)
        self._lock = threading.Lock()

    def has_start_pose(self):
        with self._lock:
            return super().has_start_pose()

    def is_ready(self):
        with self._lock:
            return super.is_ready()

    def insert(self, element, transform_func=None, start_check_func=None):
        with self._lock:
            super().insert(element, transform_func, start_check_func)

    async def get_start_pose(self):
        async with self._lock:
            return super().get_start_pose()

    def get(self):
        with self._lock:
            return super.get()
