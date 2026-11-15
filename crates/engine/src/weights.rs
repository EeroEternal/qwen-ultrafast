use anyhow::{ensure, Result};
use qwen_ultrafast_cuda::{Buffer, DevicePtr};
use qwen_ultrafast_model::{Checkpoint, Config};

#[derive(Clone, Copy)]
pub struct Matrix {
    pub ptr: DevicePtr,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy)]
pub struct Nvfp4Matrix {
    pub weight: DevicePtr,
    pub scales: DevicePtr,
    pub global_scale: DevicePtr,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy)]
pub struct HyperConnectionWeights {
    pub norm: DevicePtr,
    pub down: Matrix,
    pub up: Matrix,
    pub inject: Option<Matrix>,
}

pub struct GdnWeights {
    pub qkv: Matrix,
    pub z: Matrix,
    pub a: Matrix,
    pub b: Matrix,
    pub output: Matrix,
    pub convolution: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub norm: DevicePtr,
}

pub struct QsaWeights {
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub output: Matrix,
    pub query_norm: DevicePtr,
    pub key_norm: DevicePtr,
    pub index_qk: Matrix,
    pub index_query_norm: DevicePtr,
    pub index_key_norm: DevicePtr,
}

pub enum MixerWeights {
    Gdn(GdnWeights),
    Qsa(QsaWeights),
}

pub struct ExpertWeights {
    pub id: usize,
    pub gate: Nvfp4Matrix,
    pub up: Nvfp4Matrix,
    pub down: Nvfp4Matrix,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ExpertDeviceMatrix {
    pub weight: DevicePtr,
    pub scales: DevicePtr,
    pub global_scale: DevicePtr,
}

pub struct MoeWeights {
    pub router: Matrix,
    pub shared_gate_up: Matrix,
    pub shared_down: Matrix,
    pub shared_gate: Matrix,
    pub experts: Vec<ExpertWeights>,
    pub gate_table: DevicePtr,
    pub up_table: DevicePtr,
    pub down_table: DevicePtr,
}

pub struct PleWeights {
    pub key: Matrix,
    pub value: Matrix,
    pub key_norm: DevicePtr,
    pub query_norm: DevicePtr,
    pub convolution_norm: DevicePtr,
    pub convolution: DevicePtr,
    pub multipliers: Vec<i64>,
    pub vocabulary_sizes: Vec<i64>,
    pub offsets: Vec<i64>,
}

pub struct LayerWeights {
    pub attention_mixer: HyperConnectionWeights,
    pub mixer: MixerWeights,
    pub mlp_mixer: HyperConnectionWeights,
    pub mlp: MoeWeights,
    pub ple: Option<PleWeights>,
}

pub struct Weights {
    pub embedding: Matrix,
    pub head: Matrix,
    pub final_mixer: HyperConnectionWeights,
    pub layers: Vec<LayerWeights>,
    pub bytes: usize,
    _allocations: Vec<Box<dyn std::any::Any>>,
}

struct Loader<'a> {
    checkpoint: &'a Checkpoint,
    allocations: Vec<Box<dyn std::any::Any>>,
    bytes: usize,
}

impl Loader<'_> {
    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let values = self.checkpoint.bf16(name, shape)?;
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len() * 2;
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(ptr)
    }

    fn f32(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let values = self.checkpoint.f32(name, shape)?;
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len() * 4;
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(ptr)
    }

    fn f32_bf16(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let values: Vec<f32> = self
            .checkpoint
            .bf16(name, shape)?
            .iter()
            .map(|value| value.to_f32())
            .collect();
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len() * 4;
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(ptr)
    }

    fn u8(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let values = self.checkpoint.fp8(name, shape)?;
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len();
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(ptr)
    }

    fn raw_u8(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let values = self.checkpoint.u8(name, shape)?;
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len();
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(ptr)
    }

    fn matrix(&mut self, name: &str, n: usize, k: usize) -> Result<Matrix> {
        Ok(Matrix {
            ptr: self.bf16(name, &[n, k])?,
            n,
            k,
        })
    }

    fn merged(&mut self, names: &[String], n: usize, k: usize) -> Result<Matrix> {
        let mut values = Vec::with_capacity(n * k);
        for name in names {
            values.extend(self.checkpoint.bf16(name, &[n / names.len(), k])?);
        }
        let buffer = Buffer::from_host(&values)?;
        self.bytes += values.len() * 2;
        let ptr = buffer.ptr();
        self.allocations.push(Box::new(buffer));
        Ok(Matrix { ptr, n, k })
    }

    fn nvfp4(&mut self, name: &str, n: usize, k: usize) -> Result<Nvfp4Matrix> {
        Ok(Nvfp4Matrix {
            weight: self.raw_u8(&format!("{name}.weight"), &[n, k / 2])?,
            scales: self.u8(&format!("{name}.weight_scale"), &[n, k / 16])?,
            global_scale: self.f32(&format!("{name}.weight_scale_2"), &[])?,
            n,
            k,
        })
    }

    fn expert_tables(
        &mut self,
        experts: &[ExpertWeights],
        count: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let mut gate = vec![ExpertDeviceMatrix::default(); count];
        let mut up = vec![ExpertDeviceMatrix::default(); count];
        let mut down = vec![ExpertDeviceMatrix::default(); count];
        for expert in experts {
            let convert = |matrix: Nvfp4Matrix| ExpertDeviceMatrix {
                weight: matrix.weight,
                scales: matrix.scales,
                global_scale: matrix.global_scale,
            };
            gate[expert.id] = convert(expert.gate);
            up[expert.id] = convert(expert.up);
            down[expert.id] = convert(expert.down);
        }
        let mut put = |values: &[ExpertDeviceMatrix]| -> Result<DevicePtr> {
            let buffer = Buffer::from_host(values)?;
            self.bytes += std::mem::size_of_val(values);
            let ptr = buffer.ptr();
            self.allocations.push(Box::new(buffer));
            Ok(ptr)
        };
        Ok((put(&gate)?, put(&up)?, put(&down)?))
    }

    fn hyper_connection(
        &mut self,
        prefix: &str,
        inject: bool,
        config: &Config,
    ) -> Result<HyperConnectionWeights> {
        let text = &config.text_config;
        let wide = text.hidden_size * text.hc_count;
        Ok(HyperConnectionWeights {
            norm: self.bf16(&format!("{prefix}.hc_norm.weight"), &[wide])?,
            down: self.matrix(
                &format!("{prefix}.input_mix_weight_down.weight"),
                text.hc_lowrank,
                wide,
            )?,
            up: self.matrix(
                &format!("{prefix}.input_mix_weight_up.weight"),
                wide,
                text.hc_lowrank,
            )?,
            inject: inject
                .then(|| {
                    self.matrix(
                        &format!("{prefix}.block_inject_weight.weight"),
                        text.hc_count,
                        wide,
                    )
                })
                .transpose()?,
        })
    }
}

impl Weights {
    pub fn load(
        checkpoint: &Checkpoint,
        config: &Config,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        ensure!(
            world_size > 0 && rank < world_size,
            "invalid expert-parallel rank"
        );
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let wide = hidden * text.hc_count;
        let key_dim = text.linear_num_key_heads * text.linear_key_head_dim;
        let value_dim = text.linear_num_value_heads * text.linear_value_head_dim;
        let mut loader = Loader {
            checkpoint,
            allocations: Vec::new(),
            bytes: 0,
        };
        let embedding = loader.matrix(
            "model.language_model.embed_tokens.weight",
            text.vocab_size,
            hidden,
        )?;
        let head = loader.matrix("lm_head.weight", text.vocab_size, hidden)?;
        let final_mixer = loader.hyper_connection(
            "model.language_model.hyper_connection_mixer",
            false,
            config,
        )?;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for index in 0..text.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{index}");
            let attention_mixer = loader.hyper_connection(
                &format!("{prefix}.attn_hyper_connection"),
                true,
                config,
            )?;
            let mlp_mixer =
                loader.hyper_connection(&format!("{prefix}.mlp_hyper_connection"), true, config)?;
            let mixer = if text.layer_types[index] == "linear_attention" {
                let linear = format!("{prefix}.linear_attn");
                MixerWeights::Gdn(GdnWeights {
                    qkv: loader.matrix(
                        &format!("{linear}.in_proj_qkv.weight"),
                        2 * key_dim + value_dim,
                        hidden,
                    )?,
                    z: loader.matrix(&format!("{linear}.in_proj_z.weight"), value_dim, hidden)?,
                    a: loader.matrix(
                        &format!("{linear}.in_proj_a.weight"),
                        text.linear_num_value_heads,
                        hidden,
                    )?,
                    b: loader.matrix(
                        &format!("{linear}.in_proj_b.weight"),
                        text.linear_num_value_heads,
                        hidden,
                    )?,
                    output: loader.matrix(
                        &format!("{linear}.out_proj.weight"),
                        hidden,
                        value_dim,
                    )?,
                    convolution: loader.bf16(
                        &format!("{linear}.conv1d.weight"),
                        &[2 * key_dim + value_dim, 1, text.linear_conv_kernel_dim],
                    )?,
                    a_log: loader
                        .f32_bf16(&format!("{linear}.A_log"), &[text.linear_num_value_heads])?,
                    dt_bias: loader
                        .f32_bf16(&format!("{linear}.dt_bias"), &[text.linear_num_value_heads])?,
                    norm: loader.bf16(
                        &format!("{linear}.norm.weight"),
                        &[text.linear_value_head_dim],
                    )?,
                })
            } else {
                let attention = format!("{prefix}.self_attn");
                MixerWeights::Qsa(QsaWeights {
                    query: loader.matrix(
                        &format!("{attention}.q_proj.weight"),
                        2 * text.num_attention_heads * text.head_dim,
                        hidden,
                    )?,
                    key: loader.matrix(
                        &format!("{attention}.k_proj.weight"),
                        text.num_key_value_heads * text.head_dim,
                        hidden,
                    )?,
                    value: loader.matrix(
                        &format!("{attention}.v_proj.weight"),
                        text.num_key_value_heads * text.head_dim,
                        hidden,
                    )?,
                    output: loader.matrix(
                        &format!("{attention}.o_proj.weight"),
                        hidden,
                        text.num_attention_heads * text.head_dim,
                    )?,
                    query_norm: loader
                        .bf16(&format!("{attention}.q_norm.weight"), &[text.head_dim])?,
                    key_norm: loader
                        .bf16(&format!("{attention}.k_norm.weight"), &[text.head_dim])?,
                    index_qk: loader.matrix(
                        &format!("{attention}.indexer.index_qk_proj.weight"),
                        (text.indexer_n_heads + text.indexer_kv_heads) * text.indexer_head_dim,
                        hidden,
                    )?,
                    index_query_norm: loader.bf16(
                        &format!("{attention}.indexer.q_layernorm.weight"),
                        &[text.indexer_head_dim],
                    )?,
                    index_key_norm: loader.bf16(
                        &format!("{attention}.indexer.k_layernorm.weight"),
                        &[text.indexer_head_dim],
                    )?,
                })
            };
            let mlp = format!("{prefix}.mlp");
            let mut experts = Vec::new();
            for id in (rank..text.num_experts).step_by(world_size) {
                let expert = format!("{mlp}.experts.{id}");
                experts.push(ExpertWeights {
                    id,
                    gate: loader.nvfp4(
                        &format!("{expert}.gate_proj"),
                        text.moe_intermediate_size,
                        hidden,
                    )?,
                    up: loader.nvfp4(
                        &format!("{expert}.up_proj"),
                        text.moe_intermediate_size,
                        hidden,
                    )?,
                    down: loader.nvfp4(
                        &format!("{expert}.down_proj"),
                        hidden,
                        text.moe_intermediate_size,
                    )?,
                });
            }
            let (gate_table, up_table, down_table) =
                loader.expert_tables(&experts, text.num_experts)?;
            let mlp = MoeWeights {
                router: loader.matrix(&format!("{mlp}.gate.weight"), text.num_experts, hidden)?,
                gate_table,
                up_table,
                down_table,
                shared_gate_up: loader.merged(
                    &[
                        format!("{mlp}.shared_expert.gate_proj.weight"),
                        format!("{mlp}.shared_expert.up_proj.weight"),
                    ],
                    2 * text.shared_expert_intermediate_size,
                    hidden,
                )?,
                shared_down: loader.matrix(
                    &format!("{mlp}.shared_expert.down_proj.weight"),
                    hidden,
                    text.shared_expert_intermediate_size,
                )?,
                shared_gate: loader.matrix(
                    &format!("{mlp}.shared_expert_gate.weight"),
                    1,
                    hidden,
                )?,
                experts,
            };
            let ple = if text.ple_layer_ids.contains(&(index + 1)) {
                let ple = format!("{prefix}.ple");
                Some(PleWeights {
                    key: loader.matrix(
                        &format!("{ple}.key_proj.weight"),
                        wide,
                        text.ple_embed_dim,
                    )?,
                    value: loader.matrix(
                        &format!("{ple}.value_proj.weight"),
                        hidden,
                        text.ple_embed_dim,
                    )?,
                    key_norm: loader.bf16(&format!("{ple}.norm_key.weight"), &[wide])?,
                    query_norm: loader.bf16(&format!("{ple}.norm_query.weight"), &[wide])?,
                    convolution_norm: loader.bf16(&format!("{ple}.norm_conv.weight"), &[wide])?,
                    convolution: loader.bf16(
                        &format!("{ple}.conv1d.weight"),
                        &[wide, 1, text.ple_conv_kernel_size],
                    )?,
                    multipliers: checkpoint.i64(
                        &format!("{ple}.ple_embedding.layer_multipliers"),
                        &[text.ngram_size],
                    )?,
                    vocabulary_sizes: checkpoint.i64(
                        &format!("{ple}.ple_embedding.ngram_heads_vocab_sizes"),
                        &[(text.ngram_size - 1) * text.heads_per_ngram],
                    )?,
                    offsets: checkpoint.i64(
                        &format!("{ple}.ple_embedding.ngram_heads_offsets"),
                        &[(text.ngram_size - 1) * text.heads_per_ngram],
                    )?,
                })
            } else {
                None
            };
            layers.push(LayerWeights {
                attention_mixer,
                mixer,
                mlp_mixer,
                mlp,
                ple,
            });
        }
        Ok(Self {
            embedding,
            head,
            final_mixer,
            layers,
            bytes: loader.bytes,
            _allocations: loader.allocations,
        })
    }
}
