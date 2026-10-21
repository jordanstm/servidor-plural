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

#[cfg(not(debug_assertions))]
fn get_embedded_paths() -> (std::path::PathBuf, String) {
    let temp_dir = std::env::temp_dir().join("ServidorPluralTemp");
    let _ = std::fs::create_dir_all(&temp_dir);

    let use_cuda = std::process::Command::new("nvidia-smi").output().is_ok();
    let server_filename = if use_cuda { "servidor_cuda.exe" } else { "servidor_cpu.exe" };
    
    let server_path = temp_dir.join(server_filename);
    
    // Sempre reescrever o servidor para garantir a versão correta se ele trocar de GPU
    let bytes: &[u8] = if use_cuda {
        include_bytes!("../../../target/release/servidor_cuda.exe")
    } else {
        include_bytes!("../../../target/release/servidor_cpu.exe")
    };
    let _ = std::fs::write(&server_path, bytes);

    let compressor_path = temp_dir.join("CompressorVRAM.exe");
    if !compressor_path.exists() {
        let _ = std::fs::write(&compressor_path, include_bytes!("../../../CompressorVRAM.exe"));
    }
    
    let dll_path = temp_dir.join("vram_allocator.dll");
    if !dll_path.exists() {
        let _ = std::fs::write(&dll_path, include_bytes!("../../../vram_allocator.dll"));
    }

    (temp_dir, server_filename.to_string())
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
    #[cfg(debug_assertions)]
    let server_path = base_path.join("target").join("release").join("servidor_plural_apps_sem_mic.exe");
    
    #[cfg(not(debug_assertions))]
    let (base_path, server_filename) = get_embedded_paths();
    #[cfg(not(debug_assertions))]
    let server_path = base_path.join(server_filename);
    
    let log_path = std::path::PathBuf::from(format!("{}\\Desktop\\ServidorPlural_Log.txt", std::env::var("USERPROFILE").unwrap_or_else(|_| "C:".to_string())));
    let mut cmd = Command::new(&server_path);
    cmd.current_dir(&base_path);
    
    if let Ok(log_file) = std::fs::File::create(&log_path) {
        if let Ok(log_err) = log_file.try_clone() {
            cmd.stdout(std::process::Stdio::from(log_file));
            cmd.stderr(std::process::Stdio::from(log_err));
        }
    }

    if let Ok(new_child) = cmd.spawn() {
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
            #[cfg(debug_assertions)]
            let compressor_path = base_path.join("CompressorVRAM.exe");
            #[cfg(debug_assertions)]
            let server_path = base_path.join("target").join("release").join("servidor_plural_apps_sem_mic.exe");
            
            #[cfg(not(debug_assertions))]
            let (base_path, server_filename) = get_embedded_paths();
            #[cfg(not(debug_assertions))]
            let compressor_path = base_path.join("CompressorVRAM.exe");
            #[cfg(not(debug_assertions))]
            let server_path = base_path.join(server_filename);

            println!("Iniciando CompressorVRAM: {:?}", compressor_path);
            let compressor_process = Command::new(&compressor_path)
                .current_dir(&base_path)
                .spawn()
                .ok();

            println!("Iniciando Servidor: {:?}", server_path);
            let log_path = std::path::PathBuf::from(format!("{}\\Desktop\\ServidorPlural_Log.txt", std::env::var("USERPROFILE").unwrap_or_else(|_| "C:".to_string())));
            let mut cmd = Command::new(&server_path);
            cmd.current_dir(&base_path);
            
            if let Ok(log_file) = std::fs::File::create(&log_path) {
                if let Ok(log_err) = log_file.try_clone() {
                    cmd.stdout(std::process::Stdio::from(log_file));
                    cmd.stderr(std::process::Stdio::from(log_err));
                }
            }

            let server_process = cmd.spawn().ok();

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
