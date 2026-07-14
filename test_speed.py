import asyncio
import websockets
import json
import time
import sys

async def measure_speed():
    uri = "ws://127.0.0.1:11438/stream"
    req = {
        "prompt": "Write a long story about a dragon. Write at least 400 words.",
        "max_tokens": 150,
        "temperature": 0.1
    }
    print("Connecting to ws...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected. Sending request...")
            await websocket.send(json.dumps(req))
            start_time = time.time()
            first_token_time = None
            token_count = 0
            
            while True:
                response = await websocket.recv()
                if first_token_time is None:
                    first_token_time = time.time()
                
                if response == "[DONE]":
                    break
                token_count += 1
                sys.stdout.write(response)
                sys.stdout.flush()
            
            print("\n")
            end_time = time.time()
            total_time = end_time - start_time
            gen_time = end_time - first_token_time
            tps_total = token_count / total_time if total_time > 0 else 0
            
            print(f"Total time: {total_time:.2f}s")
            print(f"Time to first token: {first_token_time - start_time:.2f}s")
            print(f"Tokens generated: {token_count}")
            print(f"Total TPS: {tps_total:.2f} t/s")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(measure_speed())
