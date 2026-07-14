@echo off
title Servidor Plural Apps (DeepSeek R1 + CompressorVram Enterprise)
echo =================================================================
echo    INICIANDO MOTOR DE LLM MEGA (DeepSeek R1 com VRAM Swap)
echo =================================================================

echo [+] 1. Iniciando o NOVO CompressorVRAM Enterprise (Tauri) em segundo plano...
:: Inicia a interface gráfica nativa já compilada
start "CompressorVRAM Tauri" "CompressorVRAM.exe"

:: Aguarda a interface do Egui subir e mapear a memória
timeout /t 4 /nobreak >nul

cd /d "c:\Projetos\ServidorPluralAppsSemMic"

echo [+] 3. Verificando/Carregando DeepSeek R1...
:: NOTA: Como você quer o DeepSeek R1, assumi o modelo destilado Llama-8B Q4 para a sua RTX 3050.
:: Coloque o caminho do GGUF correto aqui! Se nao tiver, você precisa baixar.
set MODEL_PATH=models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf

if not exist "%MODEL_PATH%" (
    echo [!] ERRO: O modelo %MODEL_PATH% nao foi encontrado!
    echo [!] Baixe o GGUF do DeepSeek R1 e coloque na pasta models/
    pause
    exit /b
)

echo [+] Iniciando Servidor Rust Plural Apps Nativo (LLAMA-CPP-2)...
echo [+] Offload de VRAM gerenciado pelo seu Driver Nativo + Modem
cargo run --release

pause
