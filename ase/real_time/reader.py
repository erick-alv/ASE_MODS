import asyncio
import time


async def follow(file):
    file.seek(0, 2)
    while True:
        line = file.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        else:
            yield line
            await asyncio.sleep(0)


async def read_file(stop_event, line_func=None, line_func_awaitable=None):
    filename = '/home/erick/Documents/UnityProjects/UbiiDemo/received.txt'
    print("Beginning reading")
    with open(filename, "r") as f:
        lines = follow(f)
        start_time = time.time()
        async for l in lines:
            end_time = time.time()
            #print(f"in reader {end_time - start_time}")#measures the time that data is read
            #print(l)
            assert line_func is None or line_func_awaitable is None, "pass just one function"
            if line_func is not None:
                line_func(l)
            elif line_func_awaitable is not None:
                await line_func_awaitable(l)

            start_time = time.time()

    if stop_event is not None:
        stop_event.set()


#todo once debug is done use real one


# async def follow():
#     counter = 0
#     while True:
#         line0 = f"[ 0.232, 1.72, 0.864, 0, 0.382683426, 0, 0.923879564, -0.284, 1.42, 1.408, 0, 0.382683426, 0, 0.923879564, 0.7623301, 1.42, 0.33367, 0, 0.382683426, 0, 0.923879564, 0, 0 ]\n"
#         line1 = f"[ 0.232, 1.72, 0.864, 0, 0.382683426, 0, 0.923879564, -0.284, 1.42, 1.408, 0, 0.382683426, 0, 0.923879564, 0.7623301, 1.42, 0.33367, 0, 0.382683426, 0, 0.923879564, 1., 0 ]\n"
#         if counter % 3 == 0:
#             yield line0
#             await asyncio.sleep(0)
#         elif counter % 3 == 1:
#             yield line1
#             await asyncio.sleep(0)
#         else:
#             await asyncio.sleep(0.001)
#         counter += 1
#
#
# async def read_file(stop_event, line_func=None, line_func_awaitable=None):
#     print("Beginning reading")
#     lines = follow()
#     start_time = time.time()
#     async for l in lines:
#         #print(l)
#         assert line_func is None or line_func_awaitable is None, "pass just one function"
#         if line_func is not None:
#             line_func(l)
#         elif line_func_awaitable is not None:
#             await line_func_awaitable(l)
#         end_time = time.time()
#         print(f"reader in {end_time - start_time}")
#         start_time = time.time()
#
#     if stop_event is not None:
#         stop_event.set()



if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(read_file(None))