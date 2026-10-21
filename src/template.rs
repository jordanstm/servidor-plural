//! Chat templates por arquitetura.
//!
//! Contexto crítico: `llama_chat_apply_template` desta versão do llama.cpp
//! NÃO é um parser Jinja (ver `llama.h`: "does not use a jinja parser. It only
//! support a pre-defined list of template"). Ele tem ~40 templates hardcoded e
//! retorna -1 para qualquer outro. O wrapper em `model.rs` faz
//! `res.try_into().expect("res is negative")` -> PANIC, não erro tratável.
//!
//! Portanto os templates abaixo são construídos à mão, espelhando o
//! `tokenizer.chat_template` que shippa dentro de cada GGUF. Os formatos foram
//! confirmados por leitura direta do header dos arquivos em
//! `C:\Projetos\CompressorVram\models`:
//!
//! - Qwen2/Qwen2.5 (arch `qwen2`): ChatML puro —
//!   `<|im_start|>role\ncontent<|im_end|>\n`, geração `<|im_start|>assistant\n`
//!   (qwen2.5-coder também injeta tools dentro do bloco system e usa
//!   `<tool_response>` para resultados de ferramenta).
//! - Nemotron-H (arch `nemotron_h`): mesmo ChatML, mais os marcadores
//!   ` thinking` / ` response` (enable_thinking = true por padrão) e o
//!   generation prompt `<|im_start|>assistant\n thinking\n`.

/// Uma mensagem de conversa. Não usamos `LlamaChatMessage` porque ele só serve
/// para caminho via `apply_chat_template` (que panicaria).
pub struct Msg {
    pub role: &'static str,
    pub content: String,
}

impl Msg {
    pub fn user(content: impl Into<String>) -> Self {
        Msg { role: "user", content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Msg { role: "assistant", content: content.into() }
    }
    /// Resultado de uma ferramenta. Nos templates ChatML vira um bloco
    /// `<tool_response>` dentro de um turno `user` (como o Jinja original).
    pub fn tool(content: impl Into<String>) -> Self {
        Msg { role: "tool", content: content.into() }
    }
}

/// Família de template detectada a partir de `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    /// ChatML (`<|im_start|>role\ncontent<|im_end|>`), sem pensamento.
    /// Qwen2/Qwen2.5 — confirmado no `tokenizer.chat_template` do GGUF.
    Qwen,
    /// ChatML + markers de pensamento ` thinking`/` response` no generation
    /// prompt. Nemotron-H (nemotron_h, nemotron_h_moe). Também Qwen3.
    QwenThinking,
    /// Template genérico: delega para o padrão `role\ncontent\n`.
    Generic,
}

impl TemplateKind {
    /// Identifica a família pelo GGUF. `arch` vem de `general.architecture`.
    pub fn from_arch(arch: &str) -> TemplateKind {
        match arch {
            // nemotron_h confirmado por leitura direta do
            // `tokenizer.chat_template` do GGUF: ChatML + ` thinking`.
            "nemotron_h" | "nemotron_h_moe" | "qwen3" | "qwen3moe" | "qwen3next" => {
                TemplateKind::QwenThinking
            }
            // qwen2.5-coder confirmado por leitura direta do GGUF: ChatML puro.
            "qwen" | "qwen2" | "qwen2moe" | "qwen2vl" => TemplateKind::Qwen,
            _ => TemplateKind::Generic,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            TemplateKind::Qwen => "qwen (chatml)",
            TemplateKind::QwenThinking => "qwen-thinking (chatml + thinking)",
            TemplateKind::Generic => "generic (role\\ncontent\\n)",
        }
    }
}

/// Monta o prompt final.
///
/// `thinking` só tem efeito em `QwenThinking` e reflete o `enable_thinking`
/// do Jinja original. Quando false, abrimos e fechamos o think vazio — é
/// exatamente o que o template faz.
pub fn build(kind: TemplateKind, system: &str, history: &[Msg], thinking: bool) -> String {
    let mut p = String::with_capacity(1024);

    match kind {
        TemplateKind::Generic => {
            // System é sempre o primeiro bloco. Se vier vazio, omitimos a linha.
            if !system.trim().is_empty() {
                p.push_str("system\n");
                p.push_str(system);
                p.push('\n');
            }

            for m in history {
                // Resultado de ferramenta entra como turno de user.
                if m.role == "tool" {
                    p.push_str("user\n");
                    p.push_str(&m.content);
                    p.push('\n');
                    continue;
                }
                p.push_str(m.role);
                p.push('\n');
                p.push_str(&m.content);
                p.push('\n');
            }

            p.push_str("assistant\n");
        }
        TemplateKind::Qwen | TemplateKind::QwenThinking => {
            // ChatML: bloco `system` sempre presente (o Jinja real grava mesmo
            // vazio — `system_message = ""` quando não há role=system).
            p.push_str("<|im_start|>system\n");
            p.push_str(system);
            p.push_str("<|im_end|>\n");

            for m in history {
                if m.role == "tool" {
                    // Resultado de ferramenta: `<tool_response>` num turno user.
                    p.push_str("<|im_start|>user\n<tool_response>\n");
                    p.push_str(&m.content);
                    p.push_str("\n</tool_response>\n<|im_end|>\n");
                    continue;
                }
                p.push_str("<|im_start|>");
                p.push_str(m.role);
                p.push('\n');
                p.push_str(&m.content);
                p.push_str("<|im_end|>\n");
            }

            // Generation prompt: abre o turno do assistant.
            p.push_str("<|im_start|>assistant\n");
            if kind == TemplateKind::QwenThinking {
                if thinking {
                    p.push_str(" thinking\n");
                } else {
                    p.push_str(" thinking response");
                }
            }
        }
    }

    p
}

/// Recorta o raciocínio do output de modelos com thought (Nemotron-H/Qwen3).
///
/// O modelo escreve `{raciocínio}\n response {resposta}`: o template real
/// normaliza o histórico com `split(' response')[-1]` (ver
/// `tokenizer.chat_template`, macro de truncagem), então seguimos a mesma
/// regra. Sem marcador, o texto volta intacto (modelo respondeu direto).
pub fn strip_think(text: &str) -> String {
    // 0) Bloco `<think>...</think>` nativo (Qwen3/Nemotron): só o conteúdo
    //    depois do fechamento é resposta. Se restar vazio, cai nas regras
    //    seguintes (marcador ` response` no início, etc).
    if let Some(start) = text.find("<think>") {
        let after = &text[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            let answer = after[end + "</think>".len()..].trim();
            if !answer.is_empty() {
                return answer.to_string();
            }
        }
    }
    // 1) Raciocínio aberto pelo prompt: corta a partir do FECHAMENTO
    //    (`rfind` no último marcador, como o template faz com split()[-1]).
    if let Some(i) = text.rfind("\n response") {
        return text[i + "\n response".len()..].trim().to_string();
    }
    // 2) Modo sem-think: o modelo re-emite o marcador no início da resposta.
    if let Some(rest) = text.strip_prefix(" response") {
        return rest.trim().to_string();
    }
    text.to_string()
}