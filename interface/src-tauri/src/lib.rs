use std::process::{Command, Child};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};
use std::fs;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn list_models(path: &str) -> Result<Vec<String>, String> {
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            models.push(name.to_string());
                        }
                    }
                }
            }
        }
    } else {
        return Err("Falha ao ler o diretório".to_string());
    }
    Ok(models)
}

#[tauri::command]
fn restart_server(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut server_guard = state.server_process.lock().unwrap();
    if let Some(mut child) = server_guard.take() {
        let _ = child.kill();
    }
    
    // Respawn the server
    #[cfg(debug_assertions)]
    let base_path = {
        let mut p = std::env::current_dir().unwrap();
        if p.ends_with("src-tauri") {
            p.pop(); p.pop();
        } else if p.ends_with("interface") {
            p.pop();
        }
        p
    };
    
    #[cfg(not(debug_assertions))]
    let base_path = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    
    let server_path = base_path.join("target").join("release").join("servidor_plural_apps_sem_mic.exe");
    if let Ok(new_child) = Command::new(&server_path).current_dir(&base_path).spawn() {
        *server_guard = Some(new_child);
        Ok(())
    } else {
        Err("Falha ao reiniciar o servidor".to_string())
    }
}

struct AppState {
    server_process: Mutex<Option<Child>>,
    compressor_process: Mutex<Option<Child>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut app_builder = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            #[cfg(debug_assertions)]
            let base_path = {
                let mut p = std::env::current_dir().unwrap();
                if p.ends_with("src-tauri") {
                    p.pop();
                    p.pop();
                } else if p.ends_with("interface") {
                    p.pop();
                }
                p
            };
            
            #[cfg(not(debug_assertions))]
            let base_path = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();

            let compressor_path = base_path.join("CompressorVRAM.exe");
            let server_path = base_path.join("target").join("release").join("servidor_plural_apps_sem_mic.exe");

            println!("Iniciando CompressorVRAM: {:?}", compressor_path);
            let compressor_process = Command::new(&compressor_path)
                .current_dir(&base_path)
                .spawn()
                .ok();

            println!("Iniciando Servidor: {:?}", server_path);
            let server_process = Command::new(&server_path)
                .current_dir(&base_path)
                .spawn()
                .ok();

            app.manage(AppState {
                server_process: Mutex::new(server_process),
                compressor_process: Mutex::new(compressor_process),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet, list_models, restart_server])
        .build(tauri::generate_context!())
        .expect("error while running tauri application");

    app_builder.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            let state = app_handle.state::<AppState>();
            
            let mut server_guard = state.server_process.lock().unwrap();
            if let Some(mut child) = server_guard.take() {
                let _ = child.kill();
            }
            
            let mut compressor_guard = state.compressor_process.lock().unwrap();
            if let Some(mut child) = compressor_guard.take() {
                let _ = child.kill();
            }
        }
    });
}
