use crate::weights::Nvfp4Matrix;
use anyhow::Result;
use qwen_ultrafast_cuda::{launch, Args, DevicePtr, Launch, Stream};
use qwen_ultrafast_kernels::Kernels;

fn launch_kernel(
    function: qwen_ultrafast_cuda::sys::CUfunction,
    stream: &Stream,
    grid: (u32, u32, u32),
    block: u32,
    args: &mut Args,
) -> Result<()> {
    unsafe {
        launch(
            function,
            Launch {
                grid,
                block: (block, 1, 1),
                shared_bytes: 0,
            },
            stream,
            args,
        )
    }
}

pub fn gdn_conv(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    channels: usize,
) -> Result<()> {
    launch_kernel(
        kernels.gdn_conv,
        stream,
        (channels.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(input)
            .u64(weight)
            .u64(state)
            .u64(output)
            .u32(channels as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence(
    kernels: &Kernels,
    stream: &Stream,
    qkv: DevicePtr,
    z: DevicePtr,
    a: DevicePtr,
    b: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    norm: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    raw_output: DevicePtr,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<()> {
    launch_kernel(
        kernels.gdn_recurrence,
        stream,
        (value_heads as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(qkv)
            .u64(z)
            .u64(a)
            .u64(b)
            .u64(a_log)
            .u64(dt_bias)
            .u64(norm)
            .u64(state)
            .u64(output)
            .u64(raw_output)
            .u32(key_heads as u32)
            .u32(value_heads as u32)
            .u32(head_dim as u32)
            .f32(epsilon),
    )
}

pub fn router(
    kernels: &Kernels,
    stream: &Stream,
    logits: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    launch_kernel(
        kernels.router_topk,
        stream,
        (1, 1, 1),
        512,
        &mut Args::new()
            .u64(logits)
            .u64(ids)
            .u64(weights)
            .u32(experts as u32)
            .u32(top_k as u32),
    )
}

pub fn nvfp4_gemv(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: Nvfp4Matrix,
    output: DevicePtr,
) -> Result<()> {
    launch_kernel(
        kernels.nvfp4_gemv,
        stream,
        (weight.n as u32, 1, 1),
        128,
        &mut Args::new()
            .u64(input)
            .u64(weight.weight)
            .u64(weight.scales)
            .u64(weight.global_scale)
            .u64(output)
            .u32(weight.n as u32)
            .u32(weight.k as u32),
    )
}

pub fn clear_f32(
    kernels: &Kernels,
    stream: &Stream,
    output: DevicePtr,
    count: usize,
) -> Result<()> {
    launch_kernel(
        kernels.clear_f32,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new().u64(output).u32(count as u32),
    )
}

pub fn expert_accumulate(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: f32,
    accumulation: DevicePtr,
    count: usize,
) -> Result<()> {
    launch_kernel(
        kernels.expert_accumulate,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(input)
            .f32(weight)
            .u64(accumulation)
            .u32(count as u32),
    )
}

pub fn moe_finish(
    kernels: &Kernels,
    stream: &Stream,
    routed: DevicePtr,
    shared: DevicePtr,
    shared_gate: DevicePtr,
    output: DevicePtr,
    count: usize,
) -> Result<()> {
    launch_kernel(
        kernels.moe_finish,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(routed)
            .u64(shared)
            .u64(shared_gate)
            .u64(output)
            .u32(count as u32),
    )
}

pub fn hc_down(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    output: DevicePtr,
    count: usize,
    streams: usize,
) -> Result<()> {
    launch_kernel(
        kernels.hc_down,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(input)
            .u64(output)
            .u32(count as u32)
            .u32(streams as u32),
    )
}

pub fn hc_mix(
    kernels: &Kernels,
    stream: &Stream,
    normed: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    hidden_size: usize,
    streams: usize,
) -> Result<()> {
    launch_kernel(
        kernels.hc_mix,
        stream,
        (hidden_size.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(normed)
            .u64(up)
            .u64(output)
            .u32(hidden_size as u32)
            .u32(streams as u32),
    )
}

pub fn hc_inject(
    kernels: &Kernels,
    stream: &Stream,
    hyper_input: DevicePtr,
    block_output: DevicePtr,
    injection_weights: DevicePtr,
    hidden_size: usize,
    streams: usize,
) -> Result<()> {
    launch_kernel(
        kernels.hc_inject,
        stream,
        (hidden_size.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(hyper_input)
            .u64(block_output)
            .u64(injection_weights)
            .u32(hidden_size as u32)
            .u32(streams as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_prepare(
    kernels: &Kernels,
    stream: &Stream,
    projected_query: DevicePtr,
    projected_key: DevicePtr,
    projected_value: DevicePtr,
    query_norm: DevicePtr,
    key_norm: DevicePtr,
    query: DevicePtr,
    gate: DevicePtr,
    key_cache: DevicePtr,
    value_cache: DevicePtr,
    state: DevicePtr,
    cosines: DevicePtr,
    sines: DevicePtr,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    max_sequence: usize,
    epsilon: f32,
) -> Result<()> {
    launch_kernel(
        kernels.qsa_prepare,
        stream,
        ((query_heads + 2 * kv_heads) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(projected_query)
            .u64(projected_key)
            .u64(projected_value)
            .u64(query_norm)
            .u64(key_norm)
            .u64(query)
            .u64(gate)
            .u64(key_cache)
            .u64(value_cache)
            .u64(state)
            .u64(cosines)
            .u64(sines)
            .u32(query_heads as u32)
            .u32(kv_heads as u32)
            .u32(head_dim as u32)
            .u32(rotary_dim as u32)
            .u32(max_sequence as u32)
            .f32(epsilon),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_attention(
    kernels: &Kernels,
    stream: &Stream,
    query: DevicePtr,
    key_cache: DevicePtr,
    value_cache: DevicePtr,
    state: DevicePtr,
    partial_values: DevicePtr,
    partial_max: DevicePtr,
    partial_sum: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_sequence: usize,
    splits: usize,
    scale: f32,
) -> Result<()> {
    let mut partial_args = Args::new()
        .u64(query)
        .u64(key_cache)
        .u64(value_cache)
        .u64(state)
        .u64(partial_values)
        .u64(partial_max)
        .u64(partial_sum)
        .u32(query_heads as u32)
        .u32(kv_heads as u32)
        .u32(head_dim as u32)
        .u32(max_sequence as u32)
        .u32(splits as u32)
        .f32(scale);
    unsafe {
        launch(
            kernels.attention_partial,
            Launch {
                grid: (query_heads as u32, splits as u32, 1),
                block: (256, 1, 1),
                shared_bytes: (max_sequence.div_ceil(splits) * 4 + 16) as u32,
            },
            stream,
            &mut partial_args,
        )?;
    }
    launch_kernel(
        kernels.qsa_reduce,
        stream,
        (query_heads as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(partial_values)
            .u64(partial_max)
            .u64(partial_sum)
            .u64(gate)
            .u64(output)
            .u32(query_heads as u32)
            .u32(head_dim as u32)
            .u32(splits as u32),
    )
}

pub fn hc_expand(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    output: DevicePtr,
    hidden_size: usize,
    streams: usize,
) -> Result<()> {
    launch_kernel(
        kernels.hc_expand,
        stream,
        (hidden_size.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(input)
            .u64(output)
            .u32(hidden_size as u32)
            .u32(streams as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ple_hash(
    kernels: &Kernels,
    stream: &Stream,
    token: u32,
    history: DevicePtr,
    multipliers: DevicePtr,
    vocabulary_sizes: DevicePtr,
    offsets: DevicePtr,
    row_ids: DevicePtr,
    eos_token: u32,
) -> Result<()> {
    launch_kernel(
        kernels.ple_hash,
        stream,
        (1, 1, 1),
        32,
        &mut Args::new()
            .u32(token)
            .u64(history)
            .u64(multipliers)
            .u64(vocabulary_sizes)
            .u64(offsets)
            .u64(row_ids)
            .u32(eos_token),
    )
}

pub fn ple_gate(
    kernels: &Kernels,
    stream: &Stream,
    key: DevicePtr,
    query: DevicePtr,
    value: DevicePtr,
    output: DevicePtr,
    hidden_size: usize,
) -> Result<()> {
    launch_kernel(
        kernels.ple_gate,
        stream,
        (4, 1, 1),
        256,
        &mut Args::new()
            .u64(key)
            .u64(query)
            .u64(value)
            .u64(output)
            .u32(hidden_size as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ple_conv(
    kernels: &Kernels,
    stream: &Stream,
    gated: DevicePtr,
    normed: DevicePtr,
    weight: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    channels: usize,
) -> Result<()> {
    launch_kernel(
        kernels.ple_conv,
        stream,
        (channels.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(gated)
            .u64(normed)
            .u64(weight)
            .u64(state)
            .u64(output)
            .u32(channels as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv_indirect(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    table: DevicePtr,
    expert_ids: DevicePtr,
    slot: usize,
    output: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    launch_kernel(
        kernels.nvfp4_gemv_indirect,
        stream,
        (n as u32, 1, 1),
        128,
        &mut Args::new()
            .u64(input)
            .u64(table)
            .u64(expert_ids)
            .u32(slot as u32)
            .u64(output)
            .u32(n as u32)
            .u32(k as u32),
    )
}

pub fn expert_accumulate_indirect(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weights: DevicePtr,
    slot: usize,
    accumulation: DevicePtr,
    count: usize,
) -> Result<()> {
    launch_kernel(
        kernels.expert_accumulate_indirect,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(input)
            .u64(weights)
            .u32(slot as u32)
            .u64(accumulation)
            .u32(count as u32),
    )
}

pub fn peer_sum(
    kernels: &Kernels,
    stream: &Stream,
    peers: [DevicePtr; 8],
    world_size: usize,
    output: DevicePtr,
    count: usize,
) -> Result<()> {
    launch_kernel(
        kernels.peer_sum,
        stream,
        (count.div_ceil(256) as u32, 1, 1),
        256,
        &mut Args::new()
            .u64(peers[0])
            .u64(peers[1])
            .u64(peers[2])
            .u64(peers[3])
            .u64(peers[4])
            .u64(peers[5])
            .u64(peers[6])
            .u64(peers[7])
            .u32(world_size as u32)
            .u64(output)
            .u32(count as u32),
    )
}
