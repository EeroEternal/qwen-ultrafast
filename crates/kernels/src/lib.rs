use anyhow::Result;
use qwen_ultrafast_cuda::{sys, Module};

include!(concat!(env!("OUT_DIR"), "/cubins.rs"));

pub struct Kernels {
    _modules: Vec<Module>,
    pub embedding: sys::CUfunction,
    pub gemv_bf16: sys::CUfunction,
    pub gemv_f32: sys::CUfunction,
    pub rmsnorm: sys::CUfunction,
    pub add: sys::CUfunction,
    pub swiglu: sys::CUfunction,
    pub qkv_prepare: sys::CUfunction,
    pub attention_partial: sys::CUfunction,
    pub attention_reduce: sys::CUfunction,
    pub argmax: sys::CUfunction,
    pub advance: sys::CUfunction,
    pub gdn_conv: sys::CUfunction,
    pub gdn_recurrence: sys::CUfunction,
    pub router_topk: sys::CUfunction,
    pub nvfp4_gemv: sys::CUfunction,
    pub nvfp4_gemv_indirect: sys::CUfunction,
    pub clear_f32: sys::CUfunction,
    pub expert_accumulate: sys::CUfunction,
    pub expert_accumulate_indirect: sys::CUfunction,
    pub moe_finish: sys::CUfunction,
    pub peer_sum: sys::CUfunction,
    pub hc_down: sys::CUfunction,
    pub hc_mix: sys::CUfunction,
    pub hc_inject: sys::CUfunction,
    pub qsa_prepare: sys::CUfunction,
    pub qsa_reduce: sys::CUfunction,
    pub hc_expand: sys::CUfunction,
    pub ple_hash: sys::CUfunction,
    pub ple_gate: sys::CUfunction,
    pub ple_conv: sys::CUfunction,
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let mut modules = Vec::new();
        for (_, image) in CUBINS {
            modules.push(Module::load(image)?);
        }
        let function = |index: usize, name: &str| modules[index].function(name);
        Ok(Self {
            embedding: function(0, "qwen_embedding")?,
            gemv_bf16: function(0, "qwen_gemv_bf16")?,
            gemv_f32: function(0, "qwen_gemv_f32")?,
            rmsnorm: function(0, "qwen_rmsnorm")?,
            add: function(0, "qwen_add")?,
            swiglu: function(0, "qwen_swiglu")?,
            qkv_prepare: function(0, "qwen_qkv_prepare")?,
            attention_partial: function(0, "qwen_attention_partial")?,
            attention_reduce: function(0, "qwen_attention_reduce")?,
            argmax: function(0, "qwen_argmax")?,
            advance: function(0, "qwen_advance")?,
            gdn_conv: function(1, "qwen_gdn_conv")?,
            gdn_recurrence: function(1, "qwen_gdn_recurrence")?,
            router_topk: function(2, "qwen_router_topk")?,
            nvfp4_gemv: function(2, "qwen_nvfp4_gemv")?,
            nvfp4_gemv_indirect: function(2, "qwen_nvfp4_gemv_indirect")?,
            clear_f32: function(2, "qwen_clear_f32")?,
            expert_accumulate: function(2, "qwen_expert_accumulate")?,
            expert_accumulate_indirect: function(2, "qwen_expert_accumulate_indirect")?,
            moe_finish: function(2, "qwen_moe_finish")?,
            peer_sum: function(2, "qwen_peer_sum")?,
            hc_down: function(3, "qwen_hc_down")?,
            hc_mix: function(3, "qwen_hc_mix")?,
            hc_inject: function(3, "qwen_hc_inject")?,
            qsa_prepare: function(4, "qwen_qsa_prepare")?,
            qsa_reduce: function(4, "qwen_qsa_attention_reduce")?,
            hc_expand: function(5, "qwen_hc_expand")?,
            ple_hash: function(5, "qwen_ple_hash")?,
            ple_gate: function(5, "qwen_ple_gate")?,
            ple_conv: function(5, "qwen_ple_conv")?,
            _modules: modules,
        })
    }
}
