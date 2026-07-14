// similarity.wgsl
// Tier 3: GPU-Accelerated Semantic Search

struct Config {
    query_dim: u32,
    node_count: u32,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> query_vector: array<f32>;
@group(0) @binding(2) var<storage, read> knowledge_pool: array<f32>; // Matrix [node_count, query_dim]
@group(0) @binding(3) var<storage, read_write> results: array<f32>; // Array [node_count] of similarity scores

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= config.node_count) {
        return;
    }

    let node_idx = global_id.x;
    var score: f32 = 0.0;
    let offset = node_idx * config.query_dim;

    for (var i: u32 = 0u; i < config.query_dim; i = i + 1u) {
        score = score + (query_vector[i] * knowledge_pool[offset + i]);
    }

    results[node_idx] = score;
}
