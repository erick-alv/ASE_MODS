import asyncio
import threading


#creating a threading except hook to report when thread fails:


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
            try:
                exception = f.exception()
                if exception is not None:
                    print(f"The following exception occurred in a asyncio:\n{exception}")
            except asyncio.CancelledError:
                pass  # this one is ok since we are cancelling

        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        # wait for the thread to stop; that is the loop is stopped
        self._thread.join()
        # now eliminate any async generator and close the loop
        self._event_loop.run_until_complete(self._event_loop.shutdown_asyncgens())
        self._event_loop.close()
        print("Executed stop_async")