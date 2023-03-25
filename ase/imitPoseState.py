import asyncio

class ImitPoseState:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._buffer = []
        self._lock = asyncio.Lock()

    async def is_ready(self):
        async with self._lock:
            return len(self._buffer) == self.num_timesteps

    async def insert(self, element, transform_func=None):
        async with self._lock:
            if transform_func is not None:
                element = transform_func(element)
            self._buffer.append(element)
            if len(self._buffer) > self.num_timesteps:
                self._buffer.pop(0)

    async def get(self):
        async with self._lock:
            return self._buffer.copy()
