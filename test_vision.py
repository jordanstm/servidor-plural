import asyncio
import websockets
import json
import base64
import sys
import os

async def test_vision():
    image_path = "tabela.png"
    uri = "ws://127.0.0.1:11439"
    
    if not os.path.exists(image_path):
        print(f"[ERRO] Imagem '{image_path}' nao encontrada no diretorio local.")
        return
        
    print(f"[+] Lendo imagem '{image_path}'...")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
    # Pergunta em português para transcrever a tabela real
    prompt = (
        "Identifique a tabela da imagem e transcreva os itens exatamente como estao escritos, "
        "gerando uma tabela Markdown com as colunas correspondentes (ITEM | DESCRIÇÃO | MARCA | QTD. | UNIDADE | PREÇO UNIT. | PREÇO TOTAL)."
    )
    
    payload = {
        "prompt": prompt,
        "image_base64": image_base64,
        "max_tokens": 1024,
        "temperature": 0.1,
        "repeat_penalty": 1.1
    }
    
    print(f"[+] Conectando ao Servidor de Visao em: {uri}...")
    try:
        async with websockets.connect(uri, max_size=20 * 1024 * 1024) as websocket:
            print("[+] Conexao estabelecida com sucesso!")
            print("[+] Enviando imagem e prompt...")
            await websocket.send(json.dumps(payload))
            
            print("\n--- RESPOSTA DA IA (STREAMING) ---\n")
            
            while True:
                response = await websocket.recv()
                if response == "[DONE]":
                    print("\n\n--- FIM DA RESPOSTA ---")
                    break
                elif response.startswith("[ERRO]"):
                    print(f"\n[ERRO] {response}")
                    break
                else:
                    sys.stdout.write(response)
                    sys.stdout.flush()
                    
    except Exception as e:
        print(f"\n[ERRO] Falha na conexao ou comunicacao: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
