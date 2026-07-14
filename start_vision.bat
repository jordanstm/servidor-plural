@echo off
title Servidor Plural Apps (Visao Computacional - MiniCPM-V 2.6)
echo =================================================================
echo    INICIANDO ANEXO DE VISAO COMPUTACIONAL (MiniCPM-V 2.6 - GPU)
echo =================================================================

echo [+] Iniciando o CompressorVRAM (Modem Service) em segundo plano...
start "CompressorVRAM" cmd /k "cd /d c:\Projetos\CompressorVram\vram-modem-service && cargo run --release --bin vram-modem-service"
timeout /t 3 /nobreak >nul

cd /d "c:\Projetos\ServidorPluralAppsSemMic"

echo [+] Ativando ambiente virtual (venv)...
call venv\Scripts\activate.bat

echo [+] Iniciando Servidor de Visao na porta 11439...
echo [+] Offload para GPU (RTX 3050): 10 camadas
python vision_server.py --model models/ggml-model-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --gpu-layers 10

pause
