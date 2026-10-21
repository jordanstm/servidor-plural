# 🌐 Servidor Plural (Standalone Local LLM Server)

**High-Performance Local LLM Inference Engine, OpenAI-Compatible API & Glassmorphism Web Playground for Windows x64.**

**English** | [Português](README_PT.md)

[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-blue.svg)](#)
[![Acceleration: CUDA / UCRT](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20UCRT-green.svg)](#)
[![Memory Allocator](https://img.shields.io/badge/Memory%20Engine-CompressorVRAM-purple.svg)](https://github.com/jordanstm/CompressorVRAM)

---

## 🎯 Overview

**Servidor Plural** is a sovereign, autonomous local inference server engineered in Rust. It enables seamless execution of large language models in GGUF format with hardware acceleration (NVIDIA CUDA or multi-threaded CPU), providing:

* **High-Throughput HTTP & WebSocket Server (Axum):** Ultra-fast streaming execution on port `11438`.
* **Full OpenAI API Compatibility:** Standard `/v1/chat/completions` and `/v1/models` endpoints for plug-and-play integration with LibreChat, OpenCode, TypingMind, Cursor, or custom Python agents.
* **Embedded Glassmorphism Web UI:** Embedded zero-dependency web playground with real-time token streaming, temperature tuning, and hardware monitoring.
* **Native Desktop Client (Tauri 2 + React 19):** Modern desktop interface for visual model folder selection and backend process management.
* **Bare-Metal Tensor Stabilization:** Native integration with **[CompressorVRAM](https://github.com/jordanstm/CompressorVRAM)**, automatically detecting and loading `vram_allocator.dll` to intercept `_aligned_malloc` detours and prevent out-of-memory panics during dense model loading.

---

## 🧪 Locally Tested & Validated Models

The inference engine has been validated in production on Windows x64 workstations with NVIDIA GPU acceleration:

| Model Architecture | Weights Format | Quantization | Approximate Memory | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder 7B Instruct** | GGUF | `Q8_0` (~8.1 GB) | CUDA + VRAM Hook | ✅ Validated |
| **Qwen 2.5 Coder 7B Instruct** | GGUF | `Q4_K_M` (~4.7 GB) | CUDA + VRAM Hook | ✅ Validated |
| **NVIDIA Nemotron 3 Nano 4B** | GGUF | `Q8_0` (~4.3 GB) | CUDA + VRAM Hook | ✅ Validated |
| **NVIDIA Nemotron 3 Nano 4B** | GGUF | `Q4_K_M` (~2.5 GB) | CUDA + VRAM Hook | ✅ Validated |

---

## 🚀 Quickstart

### 1. Prerequisites
* Place your `.gguf` weights inside a local folder (e.g. `models/` or any directory).
* For tensor acceleration and heap protection, ensure `vram_allocator.dll` (from [CompressorVRAM](https://github.com/jordanstm/CompressorVRAM)) is present in the application folder or system PATH.

### 2. Launching the Server
Run the automated launcher or build with Cargo:
```bash
cargo run --release
```

Or run the precompiled binary directly:
```bash
./servidor_plural_apps_sem_mic.exe
```

### 3. Accessing the Web Playground
Open your browser at:
👉 **`http://localhost:11438`**

---

## 🔌 Connecting via OpenAI SDK (Python)

Point any OpenAI-compatible client directly to the Servidor Plural endpoint:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11438/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="Qwen2.5-Coder-7B-Instruct-Q8_0",
    messages=[
        {"role": "user", "content": "Explain how a low-level memory allocator operates in Rust."}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 🛠️ Repository Architecture

```
ServidorPlural/
├── src/
│   ├── main.rs         # HTTP & WebSocket Server (Axum)
│   ├── engine.rs       # llama.cpp Engine + vram_allocator.dll injection
│   ├── oai.rs          # OpenAI Request/Response wire types
│   ├── oai_engine.rs   # Streaming token pump & context management
│   ├── registry.rs     # GGUF header parser & architecture scanner
│   └── vram.rs         # Hardware capability detection & VRAM pipeline
├── interface/          # Desktop Client (Tauri 2 + React 19 + Vite)
├── public/             # Embedded Glassmorphism Web Assets
├── Cargo.toml          # Rust Workspace Manifest
└── start.bat           # Automated launch script
```

---

## ⚡ CompressorVRAM Integration

When started with CUDA acceleration, Servidor Plural automatically searches for `vram_allocator.dll`. Once detected, it installs dynamic function detours into the Windows C Runtime (`ucrtbase.dll`) before tensor buffers are allocated. To learn more about the 512 MB circular memory pool and the Candle tensor sparsity filter, refer to the companion project [CompressorVRAM](https://github.com/jordanstm/CompressorVRAM).

---

## 📄 License

Distributed under the [MIT License](LICENSE).

---

## 🎁 Maintenance Policy / Community Notice

> **"I did my part and gifted this project to the world. Further evolution, Linux ports, and enhancements are on you. DO NOT BUG ME FOR FREE SUPPORT — fork it and build!"**  
> *(Fiz minha parte e doei este presente para todos. Agora evoluir é com vocês: **NÃO ENCHAM O SACO!**)*
