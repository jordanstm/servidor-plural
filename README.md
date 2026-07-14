# Servidor Plural (Standalone LLM)

Bem-vindo ao **Servidor Plural**, um servidor ultra-rápido para execução local de Inteligência Artificial usando modelos GGUF.

## 🚀 Como Iniciar

1. Baixe o modelo LLM desejado no formato .gguf e coloque na pasta models/ (ou na mesma pasta do executável).
2. Dê um duplo-clique no executável ServidorPluralAppsSemMic.exe.
3. Siga o menu interativo no terminal:
   - Insira o caminho do seu modelo GGUF (ex: models/meu_modelo.gguf).
   - Escolha o modo de destilação/otimização:
     * **1:** Padrão (Foco em Precisão)
     * **2:** Alta Velocidade (Otimizado)
     * **3:** Baixa VRAM (Economia máxima)

4. Acesse **http://localhost:11438** no seu navegador para abrir o Dashboard Web e interagir com o modelo.

## 🎨 Dashboard Web (Glassmorphism)
O executável já inclui uma interface moderna e responsiva nativamente. Pelo navegador, você pode visualizar as métricas de VRAM e conversar com a IA através do Playground integrado.

---
*Powered by Rust & Axum*
