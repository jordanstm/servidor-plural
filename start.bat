@echo off
title Servidor Plural Apps (LLM)
echo ====================================================
echo Iniciando Servidor Plural Apps (Standalone LLM)
echo ====================================================

echo [+] Iniciando o CompressorVRAM (Modem Service) em segundo plano...
start "CompressorVRAM" cmd /k "cd /d c:\Projetos\CompressorVram\vram-modem-service && cargo run --release --bin vram-modem-service"
timeout /t 3 /nobreak >nul

cd /d "c:\Projetos\ServidorPluralAppsSemMic"

echo Verificando dependencias e iniciando o servidor...
cargo run --release

pause
