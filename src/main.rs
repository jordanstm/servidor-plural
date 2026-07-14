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
mod vram;

use engine::InferenceEngine;

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
    
    let port = env::var("PORT").unwrap_or_else(|_| "11438".to_string());
    let addr = format!("0.0.0.0:{}", port);

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
    println!("Acesse via Tauri ou via http://localhost:11438 no navegador");
    
    // Auto-open browser removido para não abrir o navegador duplicado ao usar o Tauri
    // let _ = std::process::Command::new("cmd").args(&["/C", "start", &format!("http://localhost:{}", port)]).spawn();

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

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
            let mut lock = state.engine.lock().unwrap();
            *lock = Some(engine);
            let mut name_lock = state.model_name.lock().unwrap();
            *name_lock = payload.model_path.clone();
            axum::Json(json!({"status": "success", "message": "Motor carregado com sucesso!"}))
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
                let repeat_penalty = req.repeat_penalty.unwrap_or(1.0);
                let stop_tokens = req.stop.clone();
                let prompt_stream = req.prompt.clone();
                
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                let state_stream = Arc::clone(&state);
                
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
                        let _ = socket.send(Message::Text("[DONE]".to_string())).await;
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
