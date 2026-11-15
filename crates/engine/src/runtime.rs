use crate::components;
use crate::ops::{self, State};
use crate::weights::{
    GdnWeights, HyperConnectionWeights, MixerWeights, MoeWeights, PleWeights, QsaWeights, Weights,
};
use anyhow::{ensure, Result};
use half::bf16;
use qwen_ultrafast_cuda::{Buffer, Device, DevicePtr, GraphExec, Stream};
use qwen_ultrafast_kernels::Kernels;
use qwen_ultrafast_model::{Checkpoint, Config};
use qwen_ultrafast_nccl::Comm;

const SPLITS: usize = 16;
const PLE_ROWS_PER_PLANE: usize = 2_500_012;

pub type Reducer<'a> = &'a mut dyn FnMut(&Kernels, &Stream, &Buffer<f32>) -> Result<()>;

pub struct Runtime<'a> {
    _device: Device,
    pub config: Config,
    pub weights: Weights,
    checkpoint: &'a Checkpoint,
    rank: usize,
    stream: Stream,
    kernels: Kernels,
    state: Buffer<State>,
    next_token: Buffer<u32>,
    pub output_tokens: Buffer<u32>,
    embedding: Buffer<bf16>,
    hyper: Buffer<bf16>,
    normed: Buffer<bf16>,
    down: Buffer<bf16>,
    up: Buffer<bf16>,
    mixed: Buffer<bf16>,
    injection: Buffer<bf16>,
    block: Buffer<bf16>,
    projected: Buffer<bf16>,
    qkv: Buffer<bf16>,
    z: Buffer<bf16>,
    a: Buffer<bf16>,
    b: Buffer<bf16>,
    query: Buffer<bf16>,
    gate: Buffer<bf16>,
    key: Buffer<bf16>,
    value: Buffer<bf16>,
    gate_up: Buffer<bf16>,
    intermediate: Buffer<bf16>,
    router_logits: Buffer<bf16>,
    expert_ids: Buffer<u32>,
    expert_weights: Buffer<f32>,
    routed: Buffer<f32>,
    shared_gate: Buffer<f32>,
    partial_values: Buffer<f32>,
    partial_max: Buffer<f32>,
    partial_sum: Buffer<f32>,
    logits: Buffer<f32>,
    cosines: Buffer<f32>,
    sines: Buffer<f32>,
    gdn_states: Vec<(usize, Buffer<bf16>, Buffer<f32>)>,
    qsa_caches: Vec<(usize, Buffer<bf16>, Buffer<bf16>)>,
    ple_embedding: Buffer<bf16>,
    ple_key: Buffer<bf16>,
    ple_value: Buffer<bf16>,
    ple_gated: Buffer<bf16>,
    ple_normed: Buffer<bf16>,
    ple_output: Buffer<bf16>,
    ple_rows: Buffer<u64>,
    ple_token_history: Buffer<u32>,
    ple_convolution: Buffer<bf16>,
    ple_multipliers: Buffer<u64>,
    ple_vocabulary_sizes: Buffer<u64>,
    ple_offsets: Buffer<u64>,
    ple_global_scale: f32,
}

impl<'a> Runtime<'a> {
    pub fn new(
        ordinal: i32,
        rank: usize,
        world_size: usize,
        checkpoint: &'a Checkpoint,
        config: Config,
        max_sequence: usize,
    ) -> Result<Self> {
        ensure!(
            max_sequence > 0 && max_sequence <= 2048,
            "the dense-attention MVP supports 1..=2048 tokens"
        );
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let wide = hidden * text.hc_count;
        let key_dim = text.linear_num_key_heads * text.linear_key_head_dim;
        let value_dim = text.linear_num_value_heads * text.linear_value_head_dim;
        let query_dim = text.num_attention_heads * text.head_dim;
        let kv_dim = text.num_key_value_heads * text.head_dim;
        let device = Device::new(ordinal)?;
        let weights = Weights::load(checkpoint, &config, rank, world_size)?;
        let stream = Stream::new()?;
        let kernels = Kernels::load()?;
        let mut gdn_states = Vec::new();
        let mut qsa_caches = Vec::new();
        for (index, kind) in text.layer_types.iter().enumerate() {
            if kind == "linear_attention" {
                gdn_states.push((
                    index,
                    Buffer::from_host(&vec![bf16::ZERO; (2 * key_dim + value_dim) * 3])?,
                    Buffer::from_host(&vec![
                        0.0f32;
                        text.linear_num_value_heads
                            * text.linear_key_head_dim
                            * text.linear_value_head_dim
                    ])?,
                ));
            } else {
                qsa_caches.push((
                    index,
                    Buffer::alloc(kv_dim * max_sequence)?,
                    Buffer::alloc(kv_dim * max_sequence)?,
                ));
            }
        }
        let mut cosines = vec![0.0f32; max_sequence * text.head_dim / 8];
        let mut sines = cosines.clone();
        let rotary_dim =
            (text.head_dim as f32 * text.rope_parameters.partial_rotary_factor) as usize;
        for position in 0..max_sequence {
            for d in 0..rotary_dim / 2 {
                let angle = position as f32
                    * text
                        .rope_parameters
                        .rope_theta
                        .powf(-2.0 * d as f32 / rotary_dim as f32);
                cosines[position * rotary_dim / 2 + d] = angle.cos();
                sines[position * rotary_dim / 2 + d] = angle.sin();
            }
        }
        let ple = text
            .ple_layer_ids
            .first()
            .map(|id| &weights.layers[id - 1].ple);
        let (multipliers, vocabulary_sizes, offsets) = if let Some(Some(ple)) = ple {
            (
                ple.multipliers
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<Vec<_>>(),
                ple.vocabulary_sizes
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<Vec<_>>(),
                ple.offsets.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            )
        } else {
            (vec![1; 3], vec![1; 16], vec![0; 16])
        };
        let ple_global_scale = if text.ple_layer_ids.is_empty() {
            1.0
        } else {
            checkpoint.bf16(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
                &[1],
            )?[0]
                .to_f32()
        };
        Ok(Self {
            _device: device,
            config: config.clone(),
            weights,
            checkpoint,
            rank,
            stream,
            kernels,
            state: Buffer::from_host(&[State::default()])?,
            next_token: Buffer::alloc(1)?,
            output_tokens: Buffer::alloc(max_sequence)?,
            embedding: Buffer::alloc(hidden)?,
            hyper: Buffer::alloc(wide)?,
            normed: Buffer::alloc(wide)?,
            down: Buffer::alloc(text.hc_lowrank)?,
            up: Buffer::alloc(wide)?,
            mixed: Buffer::alloc(hidden)?,
            injection: Buffer::alloc(text.hc_count)?,
            block: Buffer::alloc(value_dim.max(query_dim))?,
            projected: Buffer::alloc(hidden)?,
            qkv: Buffer::alloc((2 * key_dim + value_dim).max(2 * query_dim))?,
            z: Buffer::alloc(value_dim)?,
            a: Buffer::alloc(text.linear_num_value_heads)?,
            b: Buffer::alloc(text.linear_num_value_heads)?,
            query: Buffer::alloc(query_dim)?,
            gate: Buffer::alloc(query_dim)?,
            key: Buffer::alloc(kv_dim)?,
            value: Buffer::alloc(kv_dim)?,
            gate_up: Buffer::alloc(
                2 * text
                    .moe_intermediate_size
                    .max(text.shared_expert_intermediate_size),
            )?,
            intermediate: Buffer::alloc(
                text.moe_intermediate_size
                    .max(text.shared_expert_intermediate_size),
            )?,
            router_logits: Buffer::alloc(text.num_experts)?,
            expert_ids: Buffer::alloc(text.num_experts_per_tok)?,
            expert_weights: Buffer::alloc(text.num_experts_per_tok)?,
            routed: Buffer::alloc(hidden)?,
            shared_gate: Buffer::alloc(1)?,
            partial_values: Buffer::alloc(SPLITS * text.num_attention_heads * text.head_dim)?,
            partial_max: Buffer::alloc(SPLITS * text.num_attention_heads)?,
            partial_sum: Buffer::alloc(SPLITS * text.num_attention_heads)?,
            logits: Buffer::alloc(text.vocab_size)?,
            cosines: Buffer::from_host(&cosines)?,
            sines: Buffer::from_host(&sines)?,
            gdn_states,
            qsa_caches,
            ple_embedding: Buffer::alloc(text.ple_embed_dim)?,
            ple_key: Buffer::alloc(wide)?,
            ple_value: Buffer::alloc(hidden)?,
            ple_gated: Buffer::alloc(wide)?,
            ple_normed: Buffer::alloc(wide)?,
            ple_output: Buffer::alloc(wide)?,
            ple_rows: Buffer::alloc(16)?,
            ple_token_history: Buffer::from_host(&[text.eos_token_id, text.eos_token_id])?,
            ple_convolution: Buffer::from_host(&vec![bf16::ZERO; wide * 9])?,
            ple_multipliers: Buffer::from_host(&multipliers)?,
            ple_vocabulary_sizes: Buffer::from_host(&vocabulary_sizes)?,
            ple_offsets: Buffer::from_host(&offsets)?,
            ple_global_scale,
        })
    }

    fn rmsnorm_streams(
        &self,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        let text = &self.config.text_config;
        for stream in 0..text.hc_count {
            let offset = (stream * text.hidden_size * 2) as u64;
            ops::rmsnorm(
                &self.kernels,
                &self.stream,
                input + offset,
                weight + offset,
                output + offset,
                text.hidden_size,
                text.rms_norm_eps,
            )?;
        }
        Ok(())
    }

    fn hyper_connection(&mut self, weights: &HyperConnectionWeights) -> Result<()> {
        let text = &self.config.text_config;
        self.rmsnorm_streams(self.hyper.ptr(), weights.norm, self.normed.ptr())?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.normed.ptr(),
            weights.down,
            self.down.ptr(),
            false,
        )?;
        components::hc_down(
            &self.kernels,
            &self.stream,
            self.down.ptr(),
            self.down.ptr(),
            text.hc_lowrank,
            text.hc_count,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.down.ptr(),
            weights.up,
            self.up.ptr(),
            false,
        )?;
        components::hc_mix(
            &self.kernels,
            &self.stream,
            self.normed.ptr(),
            self.up.ptr(),
            self.mixed.ptr(),
            text.hidden_size,
            text.hc_count,
        )?;
        if let Some(inject) = weights.inject {
            ops::gemv(
                &self.kernels,
                &self.stream,
                self.normed.ptr(),
                inject,
                self.injection.ptr(),
                false,
            )?;
        }
        Ok(())
    }

    fn inject(&mut self) -> Result<()> {
        components::hc_inject(
            &self.kernels,
            &self.stream,
            self.hyper.ptr(),
            self.projected.ptr(),
            self.injection.ptr(),
            self.config.text_config.hidden_size,
            self.config.text_config.hc_count,
        )
    }

    fn gdn(&mut self, layer: usize, weights: &GdnWeights) -> Result<()> {
        let text = &self.config.text_config;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.qkv,
            self.qkv.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.z,
            self.z.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.a,
            self.a.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.b,
            self.b.ptr(),
            false,
        )?;
        self.trace(layer, "gdn_projections")?;
        self.trace(layer, "gdn_z")?;
        self.trace(layer, "gdn_a")?;
        self.trace(layer, "gdn_b")?;
        let (_, convolution, recurrent) = self
            .gdn_states
            .iter()
            .find(|(index, _, _)| *index == layer)
            .unwrap();
        components::gdn_conv(
            &self.kernels,
            &self.stream,
            self.qkv.ptr(),
            weights.convolution,
            convolution.ptr(),
            self.qkv.ptr(),
            2 * text.linear_num_key_heads * text.linear_key_head_dim
                + text.linear_num_value_heads * text.linear_value_head_dim,
        )?;
        self.trace(layer, "gdn_convolution")?;
        components::gdn_recurrence(
            &self.kernels,
            &self.stream,
            self.qkv.ptr(),
            self.z.ptr(),
            self.a.ptr(),
            self.b.ptr(),
            weights.a_log,
            weights.dt_bias,
            weights.norm,
            recurrent.ptr(),
            self.block.ptr(),
            if std::env::var_os("QWEN_TRACE_VALUES").is_some() {
                self.query.ptr()
            } else {
                0
            },
            text.linear_num_key_heads,
            text.linear_num_value_heads,
            text.linear_value_head_dim,
            text.rms_norm_eps,
        )?;
        self.trace(layer, "gdn_raw")?;
        self.trace(layer, "gdn_recurrence")?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.block.ptr(),
            weights.output,
            self.projected.ptr(),
            false,
        )?;
        self.trace(layer, "gdn_output")
    }

    fn qsa(&mut self, layer: usize, weights: &QsaWeights) -> Result<()> {
        let text = &self.config.text_config;
        let max_sequence = self.output_tokens.len();
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.query,
            self.qkv.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.key,
            self.key.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.value,
            self.value.ptr(),
            false,
        )?;
        let (_, key_cache, value_cache) = self
            .qsa_caches
            .iter()
            .find(|(index, _, _)| *index == layer)
            .unwrap();
        components::qsa_prepare(
            &self.kernels,
            &self.stream,
            self.qkv.ptr(),
            self.key.ptr(),
            self.value.ptr(),
            weights.query_norm,
            weights.key_norm,
            self.query.ptr(),
            self.gate.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            self.state.ptr(),
            self.cosines.ptr(),
            self.sines.ptr(),
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
            (text.head_dim as f32 * text.rope_parameters.partial_rotary_factor) as usize,
            max_sequence,
            text.rms_norm_eps,
        )?;
        components::qsa_attention(
            &self.kernels,
            &self.stream,
            self.query.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            self.state.ptr(),
            self.partial_values.ptr(),
            self.partial_max.ptr(),
            self.partial_sum.ptr(),
            self.gate.ptr(),
            self.block.ptr(),
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
            max_sequence,
            SPLITS,
            1.0 / (text.head_dim as f32).sqrt(),
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.block.ptr(),
            weights.output,
            self.projected.ptr(),
            false,
        )
    }

    fn prepare_ple(&mut self, token: u32) -> Result<()> {
        if self.weights.layers.iter().all(|layer| layer.ple.is_none()) {
            return Ok(());
        }
        let text = &self.config.text_config;
        components::ple_hash(
            &self.kernels,
            &self.stream,
            token,
            self.ple_token_history.ptr(),
            self.ple_multipliers.ptr(),
            self.ple_vocabulary_sizes.ptr(),
            self.ple_offsets.ptr(),
            self.ple_rows.ptr(),
            text.eos_token_id,
        )?;
        self.stream.synchronize()?;
        let ids = self.ple_rows.download()?;
        let mut embedding = Vec::with_capacity(text.ple_embed_dim);
        for id in ids {
            let plane = (id as usize) / PLE_ROWS_PER_PLANE;
            let row = (id as usize) % PLE_ROWS_PER_PLANE;
            embedding.extend(self.checkpoint.fp8_row(
                &format!(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{plane}"
                ),
                PLE_ROWS_PER_PLANE,
                row,
                160,
                self.ple_global_scale,
            )?);
        }
        self.ple_embedding.upload(&embedding)
    }

    fn ple(&mut self, weights: &PleWeights) -> Result<()> {
        let text = &self.config.text_config;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.ple_embedding.ptr(),
            weights.key,
            self.ple_key.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.ple_embedding.ptr(),
            weights.value,
            self.ple_value.ptr(),
            false,
        )?;
        self.rmsnorm_streams(self.ple_key.ptr(), weights.key_norm, self.ple_key.ptr())?;
        self.rmsnorm_streams(self.hyper.ptr(), weights.query_norm, self.ple_normed.ptr())?;
        components::ple_gate(
            &self.kernels,
            &self.stream,
            self.ple_key.ptr(),
            self.ple_normed.ptr(),
            self.ple_value.ptr(),
            self.ple_gated.ptr(),
            text.hidden_size,
        )?;
        self.rmsnorm_streams(
            self.ple_gated.ptr(),
            weights.convolution_norm,
            self.ple_normed.ptr(),
        )?;
        components::ple_conv(
            &self.kernels,
            &self.stream,
            self.ple_gated.ptr(),
            self.ple_normed.ptr(),
            weights.convolution,
            self.ple_convolution.ptr(),
            self.ple_output.ptr(),
            text.hidden_size * text.hc_count,
        )?;
        ops::add(
            &self.kernels,
            &self.stream,
            self.hyper.ptr(),
            self.ple_output.ptr(),
            self.hyper.ptr(),
            text.hidden_size * text.hc_count,
        )
    }

    fn moe(&mut self, weights: &MoeWeights, reducer: Reducer) -> Result<()> {
        let text = &self.config.text_config;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.router,
            self.router_logits.ptr(),
            false,
        )?;
        components::router(
            &self.kernels,
            &self.stream,
            self.router_logits.ptr(),
            self.expert_ids.ptr(),
            self.expert_weights.ptr(),
            text.num_experts,
            text.num_experts_per_tok,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.shared_gate_up,
            self.gate_up.ptr(),
            false,
        )?;
        ops::swiglu(
            &self.kernels,
            &self.stream,
            self.gate_up.ptr(),
            self.intermediate.ptr(),
            text.shared_expert_intermediate_size,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.intermediate.ptr(),
            weights.shared_down,
            self.projected.ptr(),
            false,
        )?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            weights.shared_gate,
            self.shared_gate.ptr(),
            true,
        )?;
        components::clear_f32(
            &self.kernels,
            &self.stream,
            self.routed.ptr(),
            text.hidden_size,
        )?;
        for slot in 0..text.num_experts_per_tok {
            components::nvfp4_gemv_indirect(
                &self.kernels,
                &self.stream,
                self.mixed.ptr(),
                weights.gate_table,
                self.expert_ids.ptr(),
                slot,
                self.gate_up.ptr(),
                text.moe_intermediate_size,
                text.hidden_size,
            )?;
            components::nvfp4_gemv_indirect(
                &self.kernels,
                &self.stream,
                self.mixed.ptr(),
                weights.up_table,
                self.expert_ids.ptr(),
                slot,
                self.gate_up.ptr() + (text.moe_intermediate_size * 2) as u64,
                text.moe_intermediate_size,
                text.hidden_size,
            )?;
            ops::swiglu(
                &self.kernels,
                &self.stream,
                self.gate_up.ptr(),
                self.intermediate.ptr(),
                text.moe_intermediate_size,
            )?;
            components::nvfp4_gemv_indirect(
                &self.kernels,
                &self.stream,
                self.intermediate.ptr(),
                weights.down_table,
                self.expert_ids.ptr(),
                slot,
                self.block.ptr(),
                text.hidden_size,
                text.moe_intermediate_size,
            )?;
            components::expert_accumulate_indirect(
                &self.kernels,
                &self.stream,
                self.block.ptr(),
                self.expert_weights.ptr(),
                slot,
                self.routed.ptr(),
                text.hidden_size,
            )?;
        }
        reducer(&self.kernels, &self.stream, &self.routed)?;
        components::moe_finish(
            &self.kernels,
            &self.stream,
            self.routed.ptr(),
            self.projected.ptr(),
            self.shared_gate.ptr(),
            self.projected.ptr(),
            text.hidden_size,
        )
    }

    pub fn prepare_input(&mut self, token: u32, position: usize, step: usize) -> Result<()> {
        ensure!(
            position < self.output_tokens.len() && step < self.output_tokens.len(),
            "sequence capacity exceeded"
        );
        self.prepare_ple(token)?;
        self.state.upload(&[State {
            token,
            position: position as u32,
            step: step as u32,
            padding: 0,
        }])
    }

    fn trace(&self, layer: usize, stage: &str) -> Result<()> {
        if std::env::var_os("QWEN_TRACE").is_some() {
            self.stream.synchronize()?;
            eprintln!("rank {} layer {layer} {stage}", self.rank);
        }
        if self.rank == 0 && std::env::var_os("QWEN_TRACE_VALUES").is_some() {
            let buffer = match stage {
                "attention_hyper_connection" | "mlp_hyper_connection" => &self.mixed,
                "gdn_projections" | "gdn_convolution" => &self.qkv,
                "gdn_z" => &self.z,
                "gdn_a" => &self.a,
                "gdn_b" => &self.b,
                "gdn_raw" => &self.query,
                "gdn_recurrence" => &self.block,
                "mixer" | "moe" => &self.projected,
                "layer_complete" => &self.hyper,
                _ => return Ok(()),
            };
            self.stream.synchronize()?;
            let mut values = buffer.download()?;
            if stage == "gdn_projections" || stage == "gdn_convolution" {
                let text = &self.config.text_config;
                values.truncate(
                    2 * text.linear_num_key_heads * text.linear_key_head_dim
                        + text.linear_num_value_heads * text.linear_value_head_dim,
                );
            }
            let square_sum = values
                .iter()
                .map(|value| {
                    let value = value.to_f64();
                    value * value
                })
                .sum::<f64>();
            let first: Vec<f32> = values.iter().take(8).map(|value| value.to_f32()).collect();
            eprintln!(
                "layer {layer} {stage} square_sum={square_sum:.9e} first={first:?}"
            );
        }
        Ok(())
    }

    fn forward_impl(&mut self, reducer: Reducer) -> Result<()> {
        let text = self.config.text_config.clone();
        ops::embedding(
            &self.kernels,
            &self.stream,
            self.weights.embedding.ptr,
            self.state.ptr(),
            self.embedding.ptr(),
            text.hidden_size,
        )?;
        components::hc_expand(
            &self.kernels,
            &self.stream,
            self.embedding.ptr(),
            self.hyper.ptr(),
            text.hidden_size,
            text.hc_count,
        )?;
        for index in 0..self.weights.layers.len() {
            let layer = unsafe { &*self.weights.layers.as_ptr().add(index) };
            if let Some(ple) = &layer.ple {
                self.ple(ple)?;
                self.trace(index, "ple")?;
            }
            self.hyper_connection(&layer.attention_mixer)?;
            self.trace(index, "attention_hyper_connection")?;
            match &layer.mixer {
                MixerWeights::Gdn(weights) => self.gdn(index, weights)?,
                MixerWeights::Qsa(weights) => self.qsa(index, weights)?,
            }
            self.trace(index, "mixer")?;
            self.inject()?;
            self.hyper_connection(&layer.mlp_mixer)?;
            self.trace(index, "mlp_hyper_connection")?;
            self.moe(&layer.mlp, reducer)?;
            self.trace(index, "moe")?;
            self.inject()?;
            self.trace(index, "layer_complete")?;
        }
        let final_mixer = self.weights.final_mixer;
        self.hyper_connection(&final_mixer)?;
        self.trace(self.weights.layers.len(), "final_mixer")?;
        ops::gemv(
            &self.kernels,
            &self.stream,
            self.mixed.ptr(),
            self.weights.head,
            self.logits.ptr(),
            true,
        )?;
        ops::argmax(
            &self.kernels,
            &self.stream,
            self.logits.ptr(),
            self.next_token.ptr(),
            text.vocab_size,
        )?;
        ops::advance(
            &self.kernels,
            &self.stream,
            self.state.ptr(),
            self.next_token.ptr(),
            self.output_tokens.ptr(),
        )?;
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        token: u32,
        position: usize,
        step: usize,
        reducer: Reducer,
    ) -> Result<()> {
        self.prepare_input(token, position, step)?;
        self.forward_impl(reducer)?;
        self.stream.synchronize()
    }

    pub fn capture_nccl(&mut self, comm: &Comm) -> Result<GraphExec> {
        self.stream.begin_capture()?;
        self.forward_impl(&mut |_, stream, buffer| comm.all_reduce(stream, buffer))?;
        self.stream.end_capture()?.instantiate()
    }

    pub fn launch_graph(&self, graph: &GraphExec) -> Result<()> {
        graph.launch(&self.stream)?;
        self.stream.synchronize()
    }

    pub fn enable_peers(&self, contexts: &[usize]) -> Result<()> {
        for context in contexts {
            if *context != self._device.context() as usize {
                self._device.enable_peer_access(*context)?;
            }
        }
        Ok(())
    }

    pub fn routed_ptr(&self) -> DevicePtr {
        self.routed.ptr()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn synthetic_weights_size(&self) -> usize {
        self.weights.bytes
    }
}
