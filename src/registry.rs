//! Registry de modelos: escaneia `models/**/*.gguf` e lê a arquitetura
//! (`general.architecture`) direto do header do GGUF, sem depender do llama.cpp.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MODELS_DIR: &str = r"C:\Projetos\CompressorVram\models";

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub arch: String,
}

pub fn scan() -> Vec<ModelInfo> {
    let mut out = Vec::new();
    walk(Path::new(MODELS_DIR), &mut out);
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn find(id: &str) -> Option<ModelInfo> {
    let id = id.rsplit('/').next().unwrap_or(id);
    scan().into_iter().find(|m| m.id == id)
}

fn walk(dir: &Path, out: &mut Vec<ModelInfo>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
            if let Some(id) = path.file_stem().map(|s| s.to_string_lossy().to_string()) {
                let arch = read_arch(&path).unwrap_or_else(|| "unknown".to_string());
                out.push(ModelInfo { id, path, arch });
            }
        }
    }
}

/// Lê `general.architecture` do header do GGUF.
/// Layout do KV: key (u64 len + bytes) | type (u32) | valor (string = u64 len + bytes).
fn read_arch(path: &Path) -> Option<String> {
    const KEY: &[u8] = b"general.architecture";
    let mut f = File::open(path).ok()?;
    let mut head = vec![0u8; 4_000_000];
    let n = f.read(&mut head).ok()?;
    head.truncate(n);
    if head.get(0..4)? != b"GGUF" {
        return None;
    }
    let pos = find_bytes(&head, KEY)?;
    let off = pos + KEY.len() + 4;
    let len = u64::from_le_bytes(head.get(off..off + 8)?.try_into().ok()?) as usize;
    if len == 0 || len > 256 {
        return None;
    }
    let bytes = head.get(off + 8..off + 8 + len)?;
    String::from_utf8(bytes.to_vec()).ok()
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}