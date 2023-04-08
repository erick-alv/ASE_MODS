
import copy
class ImitPoseState:
    def __init__(self, num_steps_track_info):
        self.num_steps_track_info = num_steps_track_info
        self._input_buffer = []
        self.start_pose = None
        self.option = 2
        if self.option == 2:
            self.return_buffer = None
            
    def reset(self):
        self._input_buffer = []
        self.start_pose = None
        if self.option == 2:
            self.return_buffer = None
        

    # This function check if start poses has been saved;
    # this pose is used to initialize the body in the simulation
    def has_start_pose(self):
        return self.start_pose is not None

    # This function checks if the buffer has enough elements
    # so that one can pass them to the Policy
    def is_ready(self):
        if self.option == 2:
            if self.return_buffer is None:
                return len(self._input_buffer) >= self.num_steps_track_info
            else:
                return True#since we will be copying the last state

        else:
            return len(self._input_buffer) >= self.num_steps_track_info

    def insert(self, element, transform_func=None, start_check_func=None):
        if transform_func is not None:
            element = transform_func(element)
        if self.start_pose is None and start_check_func is not None:
            if(start_check_func(element)):
                self.start_pose = element
        else:
            self._input_buffer.append(element)

    def get_start_pose(self):
        return copy.deepcopy(self.start_pose)

    #should be just called if is ready is true
    def get(self):
        if self.option == 2:
            if self.return_buffer is None:
                self.return_buffer = []
                for i in range(self.num_steps_track_info):
                    self.return_buffer.append(self._input_buffer.pop(0))
            else:
                #eliminate the first one (was given in the previous call)
                self.return_buffer.pop(0)
                if len(self._input_buffer) > 0:
                    self.return_buffer.append(self._input_buffer.pop(0))
                else:
                    #copy the last state
                    self.return_buffer.append(self.return_buffer[-1])
            return self.return_buffer.copy()
        else:
            #assert self.is_ready()
            buffer = self._input_buffer[:self.num_steps_track_info].copy()
            self._input_buffer.pop(0)
            return buffer

    def getLast(self):
        if self.option == 2:
            if len(self._input_buffer) > 0:
                return self._input_buffer[-1]
            else:
                return None
        else:
            return self._input_buffer[-1]
        
        
import asyncio

class ImitPoseStateAsyncSafe(ImitPoseState):
    def __init__(self, num_steps_track_info):
        super().__init__(num_steps_track_info)
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
    def __init__(self, num_steps_track_info):
        super().__init__(num_steps_track_info)
        self._lock = threading.Lock()
        
    def reset(self):
        with self._lock:
            super().reset()

    def has_start_pose(self):
        with self._lock:
            return super().has_start_pose()

    def is_ready(self):
        with self._lock:
            return super().is_ready()

    def insert(self, element, transform_func=None, start_check_func=None):
        with self._lock:
            super().insert(element, transform_func, start_check_func)

    def get_start_pose(self):
        with self._lock:
            return super().get_start_pose()

    def get(self):
        with self._lock:
            return super().get()

    def getLast(self):
        with self._lock:
            return super().getLast()
