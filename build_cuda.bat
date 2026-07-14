@echo off
echo ========================================================
echo   Compilando Servidor Plural Apps com Suporte CUDA
echo ========================================================
echo.

:: Configurações de CUDA extraídas do projeto original (Soberano)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

:: Compilador específico do VS18
set NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe

:: Variáveis para o Llama-CPP
set LLAMA_CUDA=1
set LLAMA_CUBLAS=1

echo [+] Verificando arquitetura e compilando...
cargo build --release --features cuda

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Erro na compilação. Verifique se o CUDA v13.1 e o VS18 estão instalados.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [OK] Compilação concluída com sucesso! 
echo [OK] Binário gerado em: target\release\servidor_plural_apps_sem_mic.exe
echo.
pause
