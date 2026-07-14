import os
import ctypes

# Tenta injetar o hook do CompressorVRAM no processo atual (Python)
try:
    dll_path = os.path.abspath("vram_allocator.dll")
    if os.path.exists(dll_path):
        ctypes.CDLL(dll_path)
        print("[OK] CompressorVRAM (vram_allocator.dll) injetado com sucesso no Python!")
    else:
        print("[AVISO] vram_allocator.dll não encontrado. Hook de VRAM inativo no Python.")
except Exception as e:
    print(f"[ERRO] Falha ao injetar vram_allocator.dll: {e}")

# Desabilitar CUDA Graphs para evitar busy loops e travamento de thread no Windows
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"

import asyncio
import websockets
import json
import argparse
import threading
from typing import Optional, List, Dict, Any

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

# Configuração Padrão
HOST = "0.0.0.0"
PORT = 11439  # Rota anexa (ServidorPlural principal fica na 11438)

class VisionServer:
    def __init__(self, model_path: str, mmproj_path: str, n_gpu_layers: int = -1):
        print("--- ANEXO DE VISAO COMPUTACIONAL (MiniCPM-V 2.6 - CUDA) ---")
        self.model_path = os.path.abspath(model_path)
        self.mmproj_path = os.path.abspath(mmproj_path)
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.chat_handler = None
        self.lock = threading.Lock()
        print("[OK] Servidor de Visao MiniCPM-V Inicializado em modo SLEEP (Sem consumo de VRAM)")

    def load_model(self):
        if self.llm is None:
            print("🌅 [MOTOR] Acordando por comando... Carregando modelo na VRAM.")
            self.chat_handler = MiniCPMv26ChatHandler(clip_model_path=self.mmproj_path, verbose=False)
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_ctx=4096,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=6,
                verbose=False
            )
            print("[OK] Motor de Visão Acordado e Carregado!")

    def unload_model(self):
        if self.llm is not None:
            print("💤 [MOTOR] Hibernando... (Limpando KV Cache, mantendo VRAM)")
            try:
                self.llm.reset() # Limpa o contexto interno sem destruir o modelo na placa de vídeo
            except Exception as e:
                pass
            print("[OK] KV Cache limpo. Modelo pronto para a proxima pagina.")

    async def handle_connection(self, websocket):
        client_addr = websocket.remote_address
        print(f"[WS] Nova conexão de: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print("[WS] Erro: Falha ao parsear JSON")
                    continue

                action = data.get("action", "generate")

                if action == "sleep":
                    with self.lock:
                        self.unload_model()
                    await websocket.send("[HIBERNATED]")
                    continue

                if action == "wake":
                    with self.lock:
                        self.load_model()
                    await websocket.send("[READY]")
                    continue

                prompt = data.get("prompt", "")
                image_base64 = data.get("image_base64", "")
                max_tokens = data.get("max_tokens", 1024)
                temperature = data.get("temperature", 0.7)
                repeat_penalty = data.get("repeat_penalty", 1.1)
                
                print(f"[WS] Req recebida | Prompt len: {len(prompt)} | Tem Imagem? {'Sim' if image_base64 else 'Nao'}")

                # Tenta adquirir a trava de inferência exclusiva de forma não bloqueante
                acquired = self.lock.acquire(blocking=False)
                if not acquired:
                    print("[WS][AVISO] Rejeitando requisição: o motor de visão já está ocupando uma inferência.")
                    await websocket.send("[ERRO] O servidor de visão está ocupado processando outra tarefa. Tente novamente em instantes.")
                    await websocket.send("[DONE]")
                    continue

                # Prepara as mensagens no padrão OpenAI/MiniCPM
                messages = []
                
                content = []
                if image_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                
                content.append({"type": "text", "text": prompt})
                
                messages.append({
                    "role": "user",
                    "content": content
                })

                try:
                    self.load_model()

                    # Fila assíncrona para passar tokens da thread de geração para a thread do asyncio
                    queue = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    # Função de geração executada na thread de background
                    def generate_in_thread():
                        try:
                            stream = self.llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                repeat_penalty=repeat_penalty,
                                stream=True
                            )
                            for chunk in stream:
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    texto_gerado = delta["content"]
                                    if texto_gerado:
                                        # Envia o token de forma thread-safe para a fila do event loop do asyncio
                                        loop.call_soon_threadsafe(queue.put_nowait, texto_gerado)
                        except Exception as e:
                            print(f"[THREAD ERROR] Falha na inferência: {e}")
                            loop.call_soon_threadsafe(queue.put_nowait, f"[ERRO] {e}")
                        finally:
                            # Libera a trava exclusiva do motor de inferência
                            self.lock.release()
                            # Sinaliza o fim do streaming ao event loop do asyncio
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    # Dispara a thread nativa para a inferência não congelar o asyncio
                    inference_thread = threading.Thread(target=generate_in_thread)
                    inference_thread.start()

                    # Consome a fila assíncrona e envia os tokens via WebSocket em tempo real
                    while True:
                        token = await queue.get()
                        if token is None:
                            break
                        if token.startswith("[ERRO]"):
                            await websocket.send(token)
                            break
                        await websocket.send(token)
                    
                    # Finalizador igual ao do Rust
                    await websocket.send("[DONE]")
                    print("[WS] Resposta finalizada com sucesso.")

                except Exception as e:
                    print(f"[WS][ERRO] Falha durante inferência: {e}")
                    # Garante a liberação do lock em caso de falha antes da thread rodar
                    try:
                        self.lock.release()
                    except RuntimeError:
                        pass
                    await websocket.send(f"[ERRO] Falha na inferencia: {e}")
                    await websocket.send("[DONE]")
                finally:
                    try:
                        if self.llm is not None:
                            self.llm.reset()
                        import gc
                        gc.collect()
                        print("[WS] Recursos de inferencia e KV cache liberados na GPU.")
                    except Exception as clean_ex:
                        print(f"[WS] Erro ao limpar cache: {clean_ex}")

        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Conexão encerrada com {client_addr}")

    async def start_server(self):
        print(f"Escutando Websockets de Visão em ws://{HOST}:{PORT}")
        async with websockets.serve(self.handle_connection, HOST, PORT, ping_interval=None): # Desabilita ping timeout para conexões pesadas
            await asyncio.Future()  # Roda infinitamente

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor de Visao MiniCPM-V 2.6 (ServidorPlural)")
    parser.add_argument("--model", type=str, required=True, help="Caminho para o modelo GGUF (ex: ggml-model-Q4_K_M.gguf)")
    parser.add_argument("--mmproj", type=str, required=True, help="Caminho para o mmproj (ex: mmproj-model-f16.gguf)")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Camadas offload para GPU (-1 para usar toda a VRAM)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[ERRO] Modelo nao encontrado em {args.model}")
        exit(1)
        
    if not os.path.exists(args.mmproj):
        print(f"[ERRO] Projetor MMPROJ nao encontrado em {args.mmproj}")
        exit(1)

    server = VisionServer(model_path=args.model, mmproj_path=args.mmproj, n_gpu_layers=args.gpu_layers)
    asyncio.run(server.start_server())
