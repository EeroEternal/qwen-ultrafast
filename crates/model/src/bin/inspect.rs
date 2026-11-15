use anyhow::Result;
use qwen_ultrafast_model::{Checkpoint, Config};
use std::path::PathBuf;

fn main() -> Result<()> {
    let directory = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Qwen3.8-Flash-Next-NVFP4"));
    let config = Config::load(&directory)?;
    let checkpoint = Checkpoint::open(&directory)?;
    let text = config.text_config;
    let ple = checkpoint
        .info("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")?;
    let mut resident_bytes = 0usize;
    for name in checkpoint.tensors.keys() {
        let include = if let Some(suffix) = name.split(".experts.").nth(1) {
            let id = suffix.split('.').next().unwrap().parse::<usize>()?;
            id % 8 == 0
        } else {
            !name.starts_with("model.visual.")
                && !name.starts_with("mtp.")
                && !name.contains("ngram_embedding.shard_")
                && !name.ends_with("ngram_embedding.weight_scale")
        };
        if include {
            resident_bytes += checkpoint.bytes(name)?.len();
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_type": config.model_type,
            "layers": text.num_hidden_layers,
            "hidden_size": text.hidden_size,
            "experts_per_layer": text.num_experts,
            "tensor_count": checkpoint.tensors.len(),
            "resident_gib_per_rank_at_tp8": resident_bytes as f64 / (1u64 << 30) as f64,
            "ple_shard_dtype": ple.dtype,
            "ple_shard_shape": ple.shape,
        }))?
    );
    Ok(())
}
