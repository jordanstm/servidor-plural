use axum::{
    extract::{State, Json, ws::{WebSocketUpgrade, WebSocket, Message}},
    response::{sse::{Event, Sse}, IntoResponse},
    routing::{get, post},
    Router,
};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::Result;
use dotenvy::dotenv;
use std::env;

mod engine;
mod oai;
mod oai_engine;
mod registry;
mod template;
mod tools;
mod vram;

use engine::InferenceEngine;
use oai_engine::{Emitter, GenOptions, OaiEngine};

#[derive(Deserialize)]
struct GenerateRequest {
    action: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    repeat_penalty: Option<f32>,
    stop: Option<Vec<String>>,
}

struct AppState {
    engine: Mutex<Option<InferenceEngine>>,
    model_name: Mutex<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--cli") {
        return run_cli().await;
    }
    
    let port = env::var("PORT").unwrap_or_else(|_| "11438".to_string());
    let micelium_port = env::var("MICELIUM_PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    let oai_addr = format!("127.0.0.1:{}", micelium_port);

    let shared_state = Arc::new(AppState {
        engine: Mutex::new(None),
        model_name: Mutex::new("Não Carregado".to_string()),
    });

    let app = Router::new()
        .route("/", get(|| async { axum::response::Html(include_str!("../public/index.html")) }))
        .route("/stream", get(ws_handler))
        .route("/api/init", post(init_handler))
        .route("/api/tags", get(tags_handler))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(shared_state);

    println!("==================================================");
    println!("🌟 SERVIDOR PLURAL APPS (Standalone) 🌟");
    println!("==================================================");
    println!("Acesse via Tauri ou via http://localhost:{} no navegador", port);
    println!("");

    // ---- OpenAI-compatible layer (/v1) ----
    println!("[OAI] Inicializando backend OpenAI-compatível...");
    let oai_engine = OaiEngine::new().await?;
    let oai_state = Arc::new(oai::AppState::new(oai_engine, oai::DEFAULT_MODEL.to_string()));
    oai_state.preload();
    let oai_app = oai::router(Arc::clone(&oai_state));

    let listener_oai = tokio::net::TcpListener::bind(&oai_addr).await?;
    let listener_ui = tokio::net::TcpListener::bind(&addr).await?;

    // Listener OpenAI em thread própria (concorrente ao dashboard)
    let oai_handle = tokio::spawn(async move {
        println!("[OAI] 🐉 Servidor OpenAI ouvindo em http://{}", oai_addr);
        if let Err(e) = axum::serve(listener_oai, oai_app).await {
            eprintln!("[OAI] erro no listener: {}", e);
        }
    });

    axum::serve(listener_ui, app).await?;
    let _ = oai_handle.await;

    Ok(())
}

// ---------------------------------------------------------------------------
// CLI interativo (--cli) — usa o mesmo engine OpenAI
// ---------------------------------------------------------------------------

async fn run_cli() -> Result<()> {
    use std::io::{self, Write};
    use std::path::Path;
    use template::{Msg, TemplateKind};

    const MODEL_PATH: &str = "c:\\Projetos\\CompressorVram\\models\\Qwen2.5-Coder-7B-Instruct-Q8_0.gguf";
    const MAX_TOKENS: usize = 2048;

    let model_path = env::var("MICELIUM_MODEL")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| MODEL_PATH.to_string());
    let model_name = Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.clone());

    println!("==================================================");
    println!("🐉 SERVIDOR PLURAL DIRECT CHAT — CUDA + TIER 3");
    println!("==================================================");
    println!("  Modelo : {}", model_name);
    println!("  Swap   : CompressorVram Tier 3 Ativo");
    println!("  Diga 'sair' ou 'exit' para encerrar.\n");

    println!("[OAI] Inicializando backend...");
    let mut engine = OaiEngine::new().await?;

    let id = Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.clone());

    println!("[OAI] Carregando modelo {}...", model_path);
    let loaded = engine.ensure_path(&id, Path::new(&model_path))?;

    let arch = loaded.arch.clone();
    let kind = TemplateKind::from_arch(&arch);
    println!("[TMPL] Template: {}", kind.label());

    let thinking = match env::var("MICELIUM_THINKING") {
        Ok(v) if !v.trim().is_empty() => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "sim"
        ),
        _ => kind == TemplateKind::QwenThinking,
    };
    if kind == TemplateKind::QwenThinking {
        println!("[TMPL] Thinking: {}", if thinking { "ON" } else { "OFF" });
    }
    println!();

    let system_prompt = "Você é o ServidorPlural, um assistente de programação especialista em Rust, C++ e IA. Seja objetivo e técnico.";
    let mut history: Vec<Msg> = Vec::new();

    loop {
        print!("👤 Você: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let user_msg = input.trim();

        if user_msg.eq_ignore_ascii_case("sair") || user_msg.eq_ignore_ascii_case("exit") {
            println!("👋 Encerrando o ServidorPlural Chat. Até mais!");
            break;
        }
        if user_msg.eq_ignore_ascii_case("/limpar") || user_msg.eq_ignore_ascii_case("/clear") {
            history.clear();
            println!("🧹 Contexto limpo.\n");
            continue;
        }
        if user_msg.is_empty() {
            continue;
        }

        history.push(Msg::user(user_msg));
        let full_prompt = template::build(kind, system_prompt, &history, thinking);

        let opts = GenOptions {
            max_tokens: MAX_TOKENS,
            temperature: 0.7,
            top_p: 0.95,
            known_tools: Vec::new(),
            strip_think: true,
            greedy: true,
            thinking,
        };

        print!("🐉 ServidorPlural: ");
        io::stdout().flush()?;

        match loaded.generate(&full_prompt, &opts, &Emitter::Terminal) {
            Ok(res) => {
                println!("\n");
                history.push(Msg::assistant(res.text));
            }
            Err(e) => {
                println!("❌ Erro: {}", e);
                history.clear();
            }
        }
    }

    Ok(())
}

#[derive(Deserialize)]
struct InitRequest {
    model_path: String,
    distillation_mode: u8,
}

async fn init_handler(State(state): State<Arc<AppState>>, axum::extract::Json(payload): axum::extract::Json<InitRequest>) -> impl IntoResponse {
    println!("\n🧠 Inicializando Modelo: {}", payload.model_path);
    println!("⚙️ Modo: {}", payload.distillation_mode);
    
    match InferenceEngine::init(&payload.model_path).await {
        Ok(engine) => {
            let gpu_layers = engine.gpu_layers;
            let mut lock = state.engine.lock().unwrap();
            *lock = Some(engine);
            let mut name_lock = state.model_name.lock().unwrap();
            *name_lock = payload.model_path.clone();
            axum::Json(json!({"status": "success", "message": "Motor carregado com sucesso!", "gpu_layers": gpu_layers}))
        },
        Err(e) => {
            println!("❌ Erro ao inicializar motor: {}", e);
            axum::Json(json!({"status": "error", "message": format!("Erro: {}", e)}))
        }
    }
}



async fn tags_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let name = state.model_name.lock().unwrap().clone();
    Json(json!({
        "models": [
            {
                "name": name,
                "modified_at": "2024-03-31T00:00:00Z",
                "size": 4500000000u64,
                "digest": "sha256:micelium_tier3_engine"
            }
        ]
    }))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            if let Ok(req) = serde_json::from_str::<GenerateRequest>(&text) {
                println!("[WS] 📝 Requisição recebida | Prompt: {}...", req.prompt.chars().take(30).collect::<String>());
                
                let temperature = req.temperature.unwrap_or(0.7);
                let max_tokens = req.max_tokens.unwrap_or(2048);
                let mut repeat_penalty = req.repeat_penalty.unwrap_or(1.15);
                if repeat_penalty <= 1.0 { repeat_penalty = 1.15; }
                let stop_tokens = req.stop.clone();
                // Formatação automática baseada no nome do modelo carregado
                let mut prompt_stream = req.prompt.clone();
                let is_raw = !prompt_stream.contains("<|") && !prompt_stream.contains("[INST]") && !prompt_stream.contains("User:");
                
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                let state_stream = Arc::clone(&state);

                if is_raw {
                    let model_name_lower = state_stream.model_name.lock().unwrap().to_lowercase();
                    if model_name_lower.contains("qwen") {
                        prompt_stream = format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt_stream);
                        println!("[WS] Prompt formatado para Qwen");
                    } else if model_name_lower.contains("deepseek") {
                        // DeepSeek v2/v3 e R1 usam caracteres especiais utf-8 ｜
                        prompt_stream = format!("<｜begin of sentence｜><｜User｜>{}<｜Assistant｜>", prompt_stream);
                        println!("[WS] Prompt formatado para DeepSeek");
                    } else if model_name_lower.contains("llama-2") || model_name_lower.contains("llama2") {
                        prompt_stream = format!("[INST] {} [/INST]", prompt_stream);
                        println!("[WS] Prompt formatado para Llama 2");
                    } else {
                        // Llama 3.1 / Padrão
                        prompt_stream = format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt_stream);
                        println!("[WS] Prompt formatado para Llama 3 (Fallback Padrão)");
                    }
                }
                
                // Agora o C++ roda em uma Thread Dedicada Bloqueante fora do loop Assíncrono do Tokio!
                tokio::task::spawn_blocking(move || {
                    let mut engine_opt = state_stream.engine.lock().unwrap();
                    if let Some(engine) = engine_opt.as_mut() {
                        if let Err(e) = engine.generate_streaming(&prompt_stream, max_tokens as i32, temperature, repeat_penalty, stop_tokens, tx.clone()) {
                            println!("[ERR] Stream error: {}", e);
                        }
                    } else {
                        let _ = tx.send("[ERR] Motor não inicializado!".to_string());
                    }
                    let _ = tx.send("[DONE]".to_string());
                });
                
                while let Some(chunk) = rx.recv().await {
                    if chunk == "[DONE]" {
                        // O usuário pediu para não enviar o [DONE] para o frontend dele
                        break;
                    }
                    if !chunk.is_empty() {
                        if socket.send(Message::Text(chunk)).await.is_err() {
                            break;
                        }
                    }
                }
            } else {
                println!("[WS] Falha ao parsear JSON");
            }
        }
    }
}
