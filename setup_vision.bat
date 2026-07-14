@echo off
echo ========================================================
echo CONFIGURANDO ANEXO PYTHON (VISAO) PARA SERVIDOR PLURAL
echo ========================================================

echo 1. Criando ambiente virtual Python (venv)...
python -m venv venv
call venv\Scripts\activate.bat

echo 2. Instalando bibliotecas basicas (websockets, pydantic)...
python -m pip install --upgrade pip
pip install websockets pydantic

echo 3. Compilando e Instalando llama-cpp-python com suporte CUDA (NVIDIA)...
echo Isso pode demorar alguns minutos. Requer o Visual Studio Build Tools e CUDA Toolkit.
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo ========================================================
echo INSTALACAO CONCLUIDA! 
echo Para iniciar o servidor de visao, execute o vision_server.py
echo ========================================================
pause
