import asyncio
import websockets
import json

async def test():
    async with websockets.connect('ws://127.0.0.1:11438/stream') as ws:
        await ws.send(json.dumps({'prompt': 'Ola'}))
        while True:
            try:
                res = await ws.recv()
                print('RECV:', repr(res))
            except Exception as e:
                print('CLOSED:', e)
                break

asyncio.run(test())
