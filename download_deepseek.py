from huggingface_hub import hf_hub_download
import os

print("Baixando DeepSeek R1 (Llama-8B Distill Q4) do HuggingFace...")
os.makedirs("models", exist_ok=True)

try:
    path = hf_hub_download(
        repo_id="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        filename="DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print(f"\nDownload Concluído com Sucesso! Salvo em: {path}")
except Exception as e:
    print(f"\nErro no download: {e}")
