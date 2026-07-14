import os
import sys
from huggingface_hub import hf_hub_download

def download():
    os.makedirs("models", exist_ok=True)
    
    print("--------------------------------------------------")
    print("  BAIXANDO MODELO DE ALTA DEFINICAO MINICPM-V 2.6  ")
    print("--------------------------------------------------\n")
    
    # 1. Download do modelo principal (MiniCPM-V 2.6 Q4_K_M GGUF)
    model_filename = "ggml-model-Q4_K_M.gguf"
    print(f"[1/2] Baixando {model_filename} (Aprox. 5.3 GB)...")
    try:
        model_path = hf_hub_download(
            repo_id="openbmb/MiniCPM-V-2_6-gguf",
            filename=model_filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"[OK] Modelo salvo em: {model_path}\n")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar o modelo principal: {e}")
        sys.exit(1)
        
    # 2. Download do Projetor de Imagem (mmproj-model-f16.gguf)
    mmproj_filename = "mmproj-model-f16.gguf"
    print(f"[2/2] Baixando {mmproj_filename} (Aprox. 1.9 GB)...")
    try:
        mmproj_path = hf_hub_download(
            repo_id="openbmb/MiniCPM-V-2_6-gguf",
            filename=mmproj_filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"[OK] Projetor salvo em: {mmproj_path}\n")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar o projetor mmproj: {e}")
        sys.exit(1)
        
    print("==================================================")
    print("   TODOS OS ARQUIVOS DO MINICPM-V 2.6 BAIXADOS!")
    print("Os arquivos estao prontos na pasta './models'.")
    print("==================================================")

if __name__ == "__main__":
    download()
