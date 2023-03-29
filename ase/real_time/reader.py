import asyncio

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

async def read_file(stop_event, line_func=None):
    filename = '/home/erick/Documents/UnityProjects/UbiiDemo/received.txt'
    print("Beginning reading")
    with open(filename, "r") as f:
        lines = follow(f)
        async for l in lines:
            print(l)
            if line_func is not None:
                await line_func(l)

    if stop_event is not None:
        stop_event.set()