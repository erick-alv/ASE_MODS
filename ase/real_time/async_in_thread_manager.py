import asyncio
import threading

class AsyncInThreadManager():
    def __init__(self):
        self._event_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._event_loop.run_forever)
        self._thread.daemon = True
        self._thread.start()
        self.asyncs_list = []

    def submit_async(self, awaitable):
        print("Submitting awaitable")
        f = asyncio.run_coroutine_threadsafe(awaitable, self._event_loop)
        self.asyncs_list.append(f)
        return f

    def stop_async(self):
        for f in self.asyncs_list:
            f.cancel()
        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        #wait for the thread to stop; that is the loop is stoped and therefore all tasks in it
        self._thread.join()
        print("Called stop_async")