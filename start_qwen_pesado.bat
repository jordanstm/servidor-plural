@echo off
title Servidor Plural Apps (TESTE DE FOGO QWEN Q8)
echo =================================================================
echo    🔥 TESTE DE FOGO: MOTOR MEGA (Qwen 7B - Q8 7.7GB) 🔥
echo =================================================================

echo [+] 1. Iniciando o NOVO CompressorVRAM Enterprise (Tauri) em segundo plano...
:: Inicia a interface gráfica nativa já compilada
start "CompressorVRAM Tauri" "CompressorVRAM.exe"
timeout /t 4 /nobreak >nul

cd /d "c:\Projetos\ServidorPluralAppsSemMic"

echo [+] 3. Verificando/Carregando Qwen2.5 Coder Q8...
set MODEL_PATH=C:\Projetos\CompressorVram\models\Qwen2.5-Coder-7B-Instruct-Q8_0.gguf

if not exist "%MODEL_PATH%" (
    echo [!] ERRO: O modelo %MODEL_PATH% nao foi encontrado!
    pause
    exit /b
)

echo [+] Iniciando Servidor Rust Plural Apps Nativo (LLAMA-CPP-2)...
echo [+] Offload de VRAM e Swapping Extremo Prestes a Iniciar!
cargo run --release

pause
