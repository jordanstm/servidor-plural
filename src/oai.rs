//! Servidor HTTP compatível com a OpenAI (portado do micelium-chat).
//!
//! Endpoints:
//!   GET  /v1/models              — lista os GGUF de `models/`
//!   POST /v1/chat/completions    — completions com streaming SSE e tool calls
//!   GET  /health                 — probe de liveness
//!
//! Protocolo de tool call: o system prompt instrui o modelo a emitir um bloco
//! ```tool_call {"name":..., "arguments": {...}}```; o servidor converte num
//! `tool_calls` estilo OpenAI com `finish_reason: "tool_calls"`. A execução da
//! ferramenta é do lado do client (opencode), que devolve `role: "tool"`.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

use crate::oai_engine::{Emitter, GenOptions, GenResult, OaiEngine, Pump};
use crate::registry;
use crate::template::{self, Msg, TemplateKind};
use crate::tools::{self, ToolCallOut};

pub const DEFAULT_MODEL: &str = "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M";
const DEFAULT_MAX_TOKENS: usize = 1024;

pub struct AppState {
    pub engine: Mutex<OaiEngine>,
    pub default_model: String,
}

impl AppState {
    pub fn new(engine: OaiEngine, default_model: String) -> Self {
        AppState { engine: Mutex::new(engine), default_model }
    }

    /// Carrega o modelo default antes de subir o server (warm-up).
    pub fn preload(&self) {
        let mut eng = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        match eng.ensure(&self.default_model) {
            Ok(l) => println!("[BOOT] ✅ Modelo default pré-carregado: {}", l.id),
            Err(e) => println!("[BOOT] ⚠️ Pré-carga falhou (seguindo com lazy-load): {}", e),
        }
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::very_permissive())
        .with_state(state)
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

async fn list_models(State(_): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = registry::scan()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "servidorplural",
                "architecture": m.arch,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

// ---------------------------------------------------------------------------
// Request (formato OpenAI)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatReq {
    model: Option<String>,
    #[serde(default)]
    messages: Vec<MsgIn>,
    #[serde(default)]
    tools: Vec<tools::ToolDef>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct MsgIn {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    tool_calls: Option<Vec<TcIn>>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
struct Part {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct TcIn {
    #[serde(default)]
    function: Option<TcFn>,
}

#[derive(Deserialize)]
struct TcFn {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

impl MsgIn {
    fn text(&self) -> String {
        match &self.content {
            None => String::new(),
            Some(Content::Text(s)) => s.clone(),
            Some(Content::Parts(parts)) => parts
                .iter()
                .filter_map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatReq>,
) -> Response {
    if req.messages.is_empty() {
        return api_error(400, "`messages` vazio");
    }

    let id = format!("chatcmpl-{}", now_nanos());
    let created = now_secs();
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.default_model.clone());
    let stream = req.stream;

    let (tx, mut rx) = mpsc::unbounded_channel::<Pump>();
    {
        let state = state.clone();
        let id = id.clone();
        tokio::task::spawn_blocking(move || {
            let outcome = run_generation(&state, req, &tx, &id);
            let _ = tx.send(Pump::Done(outcome));
        });
    }

    if !stream {
        // Coleta até o Done; as peças intermediárias são ignoradas (usa o texto final).
        loop {
            match rx.recv().await {
                Some(Pump::Piece(_)) => continue,
                Some(Pump::Done(Ok(res))) => {
                    return completion_response(&id, &model_label, created, res);
                }
                Some(Pump::Done(Err(e))) => {
                    let status = if e.contains("não encontrado") { 404 } else { 500 };
                    return api_error(status, &e);
                }
                None => return api_error(500, "geração encerrada inesperadamente"),
            }
        }
    }

    // ---- Streaming SSE ----
    let chunks = async_stream::stream! {
        yield Ok::<Event, Infallible>(
            Event::default().data(chunk_json(&id, &model_label, created, json!({
                "role": "assistant", "content": ""
            }), None))
        );
        while let Some(msg) = rx.recv().await {
            match msg {
                Pump::Piece(s) => {
                    yield Ok(Event::default().data(chunk_json(
                        &id, &model_label, created,
                        json!({ "content": s }), None,
                    )));
                }
                Pump::Done(Ok(res)) => {
                    let has_tc = res.tool_call.is_some();
                    if let Some(tc) = res.tool_call {
                        yield Ok(Event::default().data(chunk_json(
                            &id, &model_label, created,
                            json!({ "tool_calls": [tool_call_delta(&tc)] }),
                            None,
                        )));
                    }
                    let finish = if has_tc { "tool_calls" } else { "stop" };
                    yield Ok(Event::default().data(chunk_json(
                        &id, &model_label, created, json!({}), Some(finish),
                    )));
                    yield Ok(Event::default().data("[DONE]"));
                }
                Pump::Done(Err(e)) => {
                    yield Ok(Event::default().data(
                        json!({ "error": { "message": e, "type": "server_error" } }).to_string()
                    ));
                    yield Ok(Event::default().data("[DONE]"));
                }
            }
        }
    };

    Sse::new(chunks).into_response()
}

// ---------------------------------------------------------------------------
// Geração (roda num thread bloqueante com o Engine travado)
// ---------------------------------------------------------------------------

fn run_generation(
    state: &AppState,
    req: ChatReq,
    tx: &mpsc::UnboundedSender<Pump>,
    _id: &str,
) -> Result<GenResult, String> {
    let mut eng = state.engine.lock().unwrap_or_else(|e| e.into_inner());

    let model_id = match req.model.as_deref() {
        Some(m) if !m.trim().is_empty() => m
            .rsplit('/')
            .next()
            .map(|s| s.to_string())
            .unwrap_or_else(|| m.to_string()),
        _ => state.default_model.clone(),
    };

    let loaded = eng.ensure(&model_id).map_err(|e| e.to_string())?;
    let n_ctx = loaded.n_ctx;
    let kind = TemplateKind::from_arch(&loaded.arch);

    // --- system (mensagens system + seção de tools) ---
    let mut system = String::new();
    let mut history: Vec<Msg> = Vec::new();
    for m in &req.messages {
        match m.role.as_str() {
            "system" | "developer" => {
                let t = m.text();
                if !t.trim().is_empty() {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(&t);
                }
            }
            "user" => history.push(Msg::user(m.text())),
            "assistant" => {
                let mut t = m.text();
                if let Some(tcs) = &m.tool_calls {
                    for tc in tcs {
                        let (name, args) = match &tc.function {
                            Some(f) => (
                                f.name.clone().unwrap_or_default(),
                                f.arguments.clone().unwrap_or_else(|| "{}".to_string()),
                            ),
                            None => (String::new(), "{}".to_string()),
                        };
                        let inner: Value = serde_json::from_str(&args)
                            .map_err(|_| "tool_calls.arguments inválido".to_string())
                            .unwrap_or(json!({}));
                        let wrapped = json!({ "name": name, "arguments": inner });
                        t.push_str("\n```tool_call\n");
                        t.push_str(&wrapped.to_string());
                        t.push_str("\n```");
                    }
                }
                if !t.trim().is_empty() {
                    history.push(Msg::assistant(t));
                }
            }
            "tool" => {
                let label = m.name.clone().unwrap_or_else(|| "tool".to_string());
                history.push(Msg::tool(format!("[resultado da ferramenta {}]\n{}", label, m.text())));
            }
            _ => {}
        }
    }
    system.push_str(&tools::system_section(&req.tools));

    // --- truncagem por orçamento de caracteres (estimativa 1 token ≈ 4 chars) ---
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .max(1);
    let budget = (n_ctx as usize).saturating_sub(max_tokens + 64) * 4;
    let mut kept: Vec<Msg> = Vec::new();
    let mut used = system.len() + 64;
    for m in history.iter().rev() {
        let cost = m.content.len() + 8;
        if used + cost > budget {
            break;
        }
        used += cost;
        kept.push(Msg { role: m.role, content: m.content.clone() });
    }
    kept.reverse();

    // --- thinking ---
    let has_tools = !req.tools.is_empty();
    let thinking = match std::env::var("MICELIUM_THINKING") {
        Ok(v) if !v.trim().is_empty() => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "sim"
        ),
        _ => kind == TemplateKind::QwenThinking && !has_tools,
    };

    let prompt = template::build(kind, &system, &kept, thinking);

    let opts = GenOptions {
        max_tokens,
        temperature: req.temperature.filter(|t| t.is_finite() && *t >= 0.0).unwrap_or(0.7),
        top_p: req.top_p.filter(|t| t.is_finite()).unwrap_or(0.95),
        known_tools: tools::known_names(&req.tools),
        strip_think: kind == TemplateKind::QwenThinking,
        greedy: false, // servidor usa amostragem por temperatura
        thinking,
    };
    eprintln!(
        "[DBG] arch={} kind={:?} strip_think={} thinking={} prompt_tail={:?}",
        loaded.arch,
        kind,
        opts.strip_think,
        opts.thinking,
        &prompt[prompt.len().saturating_sub(90)..]
    );

    let emitter = Emitter::Channel(tx.clone());
    loaded.generate(&prompt, &opts, &emitter).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Respostas
// ---------------------------------------------------------------------------

fn completion_response(
    id: &str,
    model: &str,
    created: u64,
    res: GenResult,
) -> Response {
    let mut message = json!({ "role": "assistant", "content": res.text });
    if let Some(tc) = &res.tool_call {
        message["tool_calls"] = json!([tool_call_delta(tc)]);
    }
    let finish = if res.tool_call.is_some() { "tool_calls" } else { "stop" };
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
        },
    });
    Json(body).into_response()
}

fn tool_call_delta(tc: &ToolCallOut) -> Value {
    json!({
        "index": 0,
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into()),
        }
    })
}

fn chunk_json(id: &str, model: &str, created: u64, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    })
    .to_string()
}

fn api_error(status: u16, msg: &str) -> Response {
    let code = match status {
        404 => "model_not_found",
        400 => "invalid_request_error",
        _ => "server_error",
    };
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({ "error": { "message": msg, "type": code, "code": null } })),
    )
        .into_response()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn now_nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}