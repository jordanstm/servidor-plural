# 🌐 Servidor Plural (Standalone Local LLM Server)

**High-Performance Local LLM Inference Engine, OpenAI-Compatible API & Glassmorphism Web Playground for Windows x64.**

[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-blue.svg)](#)
[![Acceleration: CUDA / UCRT](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20UCRT-green.svg)](#)
[![Memory Allocator](https://img.shields.io/badge/Memory%20Engine-CompressorVRAM-purple.svg)](https://github.com/jordanstm/CompressorVRAM)

---

## 🎯 Visão Geral

O **Servidor Plural** é uma plataforma soberana e autônoma de inferência local desenvolvida em Rust. Ela permite executar modelos de linguagem no formato GGUF com aceleração de hardware (NVIDIA CUDA ou CPU), oferecendo:

* **Servidor HTTP & WebSocket (Axum):** Execução ultra-rápida na porta `11438`.
* **API Compatível com OpenAI:** Endpoints `/v1/chat/completions` e `/v1/models` para conexão imediata com ferramentas como LibreChat, OpenCode, TypingMind ou scripts Python.
* **Dashboard Web Nativo (Glassmorphism):** Interface moderna embutida no binário com playground conversacional, controle de temperatura e monitor de hardware.
* **Aplicativo Desktop (Tauri 2 + React 19):** Interface desktop para seleção visual de diretórios de modelos e controle de ciclo de vida do servidor.
* **Coordenação Bare-Metal de Tensores:** Integração nativa com o **[CompressorVRAM](https://github.com/jordanstm/CompressorVRAM)**, utilizando a `vram_allocator.dll` para interceptar chamadas `_aligned_malloc` e prevenir estouros de heap durante a carga de matrizes densas.

---

## 🧪 Modelos Validados em Produção Local

O motor foi testado e homologado em ambiente Windows x64 com aceleração de GPU NVIDIA para os seguintes modelos:

| Modelo | Formato | Quantização | Carga de VRAM / RAM | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder 7B Instruct** | GGUF | `Q8_0` (~8.1 GB) | CUDA + VRAM Hook | ✅ Homologado |
| **Qwen 2.5 Coder 7B Instruct** | GGUF | `Q4_K_M` (~4.7 GB) | CUDA + VRAM Hook | ✅ Homologado |
| **NVIDIA Nemotron 3 Nano 4B** | GGUF | `Q8_0` (~4.3 GB) | CUDA + VRAM Hook | ✅ Homologado |
| **NVIDIA Nemotron 3 Nano 4B** | GGUF | `Q4_K_M` (~2.5 GB) | CUDA + VRAM Hook | ✅ Homologado |

---

## 🚀 Como Iniciar

### 1. Pré-requisitos
* Coloque seus arquivos `.gguf` dentro da pasta `models/` (ou em qualquer pasta local de sua preferência).
* Para aceleração de tensores e estabilização de heap, coloque a `vram_allocator.dll` (compilada a partir do [CompressorVRAM](https://github.com/jordanstm/CompressorVRAM)) na mesma pasta do executável.

### 2. Executando o Servidor
Dê um duplo-clique em `start.bat` ou execute via terminal:
```bash
cargo run --release
```

Ou execute diretamente o binário compilado:
```bash
./servidor_plural_apps_sem_mic.exe
```

### 3. Acessando a Interface
Abra seu navegador em:
👉 **`http://localhost:11438`**

---

## 🔌 Conectando como API OpenAI

Você pode apontar qualquer biblioteca compatível com a OpenAI diretamente para o Servidor Plural:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11438/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="Qwen2.5-Coder-7B-Instruct-Q8_0",
    messages=[
        {"role": "user", "content": "Explique como funciona um allocator de VRAM em Rust."}
    ]
)

print(response.choices[0].message.content)
```

---

## 🛠️ Arquitetura do Repositório

```
ServidorPlural/
├── src/
│   ├── main.rs         # Servidor HTTP/WebSocket (Axum)
│   ├── engine.rs       # Motor llama.cpp + Hook da vram_allocator.dll
│   ├── oai.rs          # Estruturas de requisição/resposta OpenAI
│   ├── oai_engine.rs   # Streaming de tokens e context cache
│   ├── registry.rs     # Varredura dinâmica de modelos .gguf
│   └── vram.rs         # Inicialização do pipeline de computação de VRAM
├── interface/          # Cliente Desktop (Tauri 2 + React 19 + Vite)
├── public/             # Assets e HTML do Dashboard Web integrado
├── Cargo.toml          # Manifesto Rust
└── start.bat           # Launcher automatizado
```

---

## ⚡ Integração com o CompressorVRAM

Quando inicializado com a flag CUDA, o Servidor Plural detecta a presença de `vram_allocator.dll` e ativa os detours de alocação de baixo nível antes de inicializar o contexto do modelo. Para saber mais sobre como o ring-buffer de 512 MB e o detour de `_aligned_malloc` operam, consulte o projeto irmão [CompressorVRAM](https://github.com/jordanstm/CompressorVRAM).

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.

---

## 🎁 Política de Manutenção / Aviso aos Navegantes

> **"Fiz minha parte e doei este presente para todos. Agora evoluir é com vocês: NÃO ENCHAM O SACO!"**  
> *(Este projeto é fornecido no estado em que se encontra. Quer suporte para Linux, novos modelos ou mais recursos? Dê um fork e implemente!)*
