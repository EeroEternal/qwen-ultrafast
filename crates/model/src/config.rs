use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub eos_token_id: u32,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: usize,
    pub rope_parameters: RopeParameters,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub indexer_head_dim: usize,
    pub indexer_kv_heads: usize,
    pub indexer_n_heads: usize,
    pub heads_per_ngram: usize,
    pub ngram_size: usize,
    pub ngram_vocab_size_base: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    pub ple_layer_ids: Vec<usize>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
}

const fn default_seed() -> u64 {
    1234
}

const fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub text_config: TextConfig,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let text = &self.text_config;
        ensure!(
            self.model_type == "qwen4_exp",
            "expected Qwen3.8 Flash Next (qwen4_exp)"
        );
        ensure!(!self.tie_word_embeddings, "expected a separate LM head");
        ensure!(
            text.hidden_size > 0 && text.num_hidden_layers > 0,
            "invalid model geometry"
        );
        ensure!(
            text.layer_types.len() == text.num_hidden_layers,
            "invalid layer type list"
        );
        ensure!(
            text.layer_types
                .iter()
                .all(|kind| matches!(kind.as_str(), "linear_attention" | "full_attention")),
            "unknown attention layer type"
        );
        ensure!(
            text.num_attention_heads % text.num_key_value_heads == 0
                && text.linear_num_value_heads % text.linear_num_key_heads == 0,
            "invalid grouped head geometry"
        );
        ensure!(
            text.head_dim <= 256 && text.linear_key_head_dim <= 256,
            "unsupported head dim"
        );
        ensure!(
            text.num_experts > 0 && text.num_experts_per_tok <= text.num_experts,
            "invalid MoE geometry"
        );
        ensure!(
            text.indexer_budget % text.indexer_compress_ratio == 0,
            "invalid QSA budget"
        );
        ensure!(
            text.ple_embed_dim % ((text.ngram_size - 1) * text.heads_per_ngram) == 0,
            "invalid PLE geometry"
        );
        Ok(())
    }
}
