//! Tool calling genérico (protocolo JSON por bloco ```tool_call).
//!
//! O opencode envia `tools[]` no request; a gente injeta o JSON Schema no
//! system prompt e espera o modelo devolver um bloco:
//!
//! ```tool_call
//! {"name": "<funcao>", "arguments": {...}}
//! ```
//!
//! O bloco é parseado e devolvido ao client como `tool_calls` +
//! `finish_reason: "tool_calls"` — é o opencode que executa a ferramenta e
//! devolve o resultado numa mensagem `role:"tool"`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: ToolFn,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolFn {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallOut {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

pub fn known_names(tools: &[ToolDef]) -> Vec<String> {
    tools.iter().map(|t| t.function.name.clone()).collect()
}

/// Seção de tools a ser concatenada ao system prompt.
pub fn system_section(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let fns: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description.clone().unwrap_or_default(),
                "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({"type":"object","properties":{}})),
            })
        })
        .collect();

    let mut s = String::new();
    s.push_str("\n\n# Tools\n\n");
    s.push_str("You have access to the following functions:\n\n");
    s.push_str(&serde_json::to_string_pretty(&fns).unwrap_or_default());
    s.push_str("\n\nWhen you need to call a function, respond with ONLY a fenced code block in exactly this format (no text before or after it):\n");
    s.push_str("```tool_call\n{\"name\": \"<function_name>\", \"arguments\": {...}}\n```\n");
    s.push_str("The `arguments` value must be a valid JSON object matching the function's parameters schema. After you receive the tool result, continue normally.\n");
    s
}

pub fn make_call_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("call_{:x}", nanos)
}

/// Faz o parse de um JSON `{"name": ..., "arguments": ...}`.
/// Se `known` não for vazio, o nome precisa estar na lista.
pub fn parse_tool_json(raw: &str, known: &[String]) -> Option<ToolCallOut> {
    let v: Value = serde_json::from_str(raw.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if !known.is_empty() && !known.iter().any(|k| k == &name) {
        return None;
    }
    let args = match v.get("arguments") {
        Some(Value::Object(_)) => v.get("arguments").cloned().unwrap(),
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        _ => json!({}),
    };
    Some(ToolCallOut { id: make_call_id(), name, arguments: args })
}

/// Fallback: procura um tool call em blocos ```/```json ou no texto inteiro,
/// pro modelos que ignoram o formato ```tool_call.
pub fn scan_embedded(text: &str, known: &[String]) -> Option<ToolCallOut> {
    if known.is_empty() {
        return None;
    }
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let lang_end = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body_start = &after[lang_end..];
        let Some(end) = body_start.find("```") else { break };
        let block = &body_start[..end];
        if let Some(tc) = parse_tool_json(block, known) {
            return Some(tc);
        }
        rest = &body_start[end + 3..];
    }
    parse_tool_json(text, known)
}