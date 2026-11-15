use crate::weights::Matrix;
use anyhow::{ensure, Result};
use qwen_ultrafast_cuda::{launch, Args, DevicePtr, Launch, Stream};
use qwen_ultrafast_kernels::Kernels;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct State {
    pub token: u32,
    pub position: u32,
    pub step: u32,
    pub padding: u32,
}

pub fn embedding(
    kernels: &Kernels,
    stream: &Stream,
    weight: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    hidden_size: usize,
) -> Result<()> {
    let mut args = Args::new()
        .u64(weight)
        .u64(state)
        .u64(output)
        .u32(hidden_size as u32);
    unsafe {
        launch(
            kernels.embedding,
            Launch {
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

pub fn gemv(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: Matrix,
    output: DevicePtr,
    float_output: bool,
) -> Result<()> {
    ensure!(
        weight.n > 0 && weight.n <= u32::MAX as usize,
        "invalid GEMV rows"
    );
    ensure!(
        weight.k > 0 && weight.k % 8 == 0,
        "GEMV K must be a multiple of eight"
    );
    let mut args = Args::new()
        .u64(input)
        .u64(weight.ptr)
        .u64(output)
        .u32(weight.n as u32)
        .u32(weight.k as u32);
    unsafe {
        launch(
            if float_output {
                kernels.gemv_f32
            } else {
                kernels.gemv_bf16
            },
            Launch {
                grid: (weight.n as u32, 1, 1),
                block: (128, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

pub fn rmsnorm(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    columns: usize,
    epsilon: f32,
) -> Result<()> {
    let mut args = Args::new()
        .u64(input)
        .u64(weight)
        .u64(output)
        .u32(columns as u32)
        .f32(epsilon);
    unsafe {
        launch(
            kernels.rmsnorm,
            Launch {
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_streams(
    kernels: &Kernels,
    stream: &Stream,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    columns: usize,
    streams: usize,
    epsilon: f32,
) -> Result<()> {
    let mut args = Args::new()
        .u64(input)
        .u64(weight)
        .u64(output)
        .u32(columns as u32)
        .f32(epsilon);
    unsafe {
        launch(
            kernels.rmsnorm_streams,
            Launch {
                grid: (streams as u32, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

pub fn add(
    kernels: &Kernels,
    stream: &Stream,
    lhs: DevicePtr,
    rhs: DevicePtr,
    output: DevicePtr,
    count: usize,
) -> Result<()> {
    let vectors = count / 8;
    let mut args = Args::new().u64(lhs).u64(rhs).u64(output).u32(count as u32);
    unsafe {
        launch(
            kernels.add,
            Launch {
                grid: (vectors.div_ceil(256) as u32, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

pub fn swiglu(
    kernels: &Kernels,
    stream: &Stream,
    gate_up: DevicePtr,
    output: DevicePtr,
    intermediate_size: usize,
) -> Result<()> {
    let mut args = Args::new()
        .u64(gate_up)
        .u64(output)
        .u32(intermediate_size as u32);
    unsafe {
        launch(
            kernels.swiglu,
            Launch {
                grid: ((intermediate_size / 8).div_ceil(256) as u32, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qkv_prepare(
    kernels: &Kernels,
    stream: &Stream,
    qkv: DevicePtr,
    q_norm: DevicePtr,
    k_norm: DevicePtr,
    query: DevicePtr,
    key_cache: DevicePtr,
    value_cache: DevicePtr,
    state: DevicePtr,
    cosines: DevicePtr,
    sines: DevicePtr,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_sequence: usize,
    epsilon: f32,
) -> Result<()> {
    let mut args = Args::new()
        .u64(qkv)
        .u64(q_norm)
        .u64(k_norm)
        .u64(query)
        .u64(key_cache)
        .u64(value_cache)
        .u64(state)
        .u64(cosines)
        .u64(sines)
        .u32(query_heads as u32)
        .u32(kv_heads as u32)
        .u32(head_dim as u32)
        .u32(max_sequence as u32)
        .f32(epsilon);
    unsafe {
        launch(
            kernels.qkv_prepare,
            Launch {
                grid: ((query_heads + 2 * kv_heads) as u32, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    kernels: &Kernels,
    stream: &Stream,
    query: DevicePtr,
    key_cache: DevicePtr,
    value_cache: DevicePtr,
    state: DevicePtr,
    partial_values: DevicePtr,
    partial_max: DevicePtr,
    partial_sum: DevicePtr,
    output: DevicePtr,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_sequence: usize,
    splits: usize,
    scale: f32,
) -> Result<()> {
    ensure!(splits > 0 && max_sequence > 0, "invalid attention geometry");
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
                shared_bytes: (max_sequence.div_ceil(splits) * 4) as u32,
            },
            stream,
            &mut partial_args,
        )?;
    }
    let mut reduce_args = Args::new()
        .u64(partial_values)
        .u64(partial_max)
        .u64(partial_sum)
        .u64(output)
        .u32(query_heads as u32)
        .u32(head_dim as u32)
        .u32(splits as u32);
    unsafe {
        launch(
            kernels.attention_reduce,
            Launch {
                grid: (query_heads as u32, 1, 1),
                block: (256, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut reduce_args,
        )
    }
}

pub fn argmax(
    kernels: &Kernels,
    stream: &Stream,
    logits: DevicePtr,
    next_token: DevicePtr,
    vocabulary: usize,
) -> Result<()> {
    let mut args = Args::new()
        .u64(logits)
        .u64(next_token)
        .u32(vocabulary as u32);
    unsafe {
        launch(
            kernels.argmax,
            Launch {
                grid: (1, 1, 1),
                block: (1024, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}

pub fn advance(
    kernels: &Kernels,
    stream: &Stream,
    state: DevicePtr,
    next_token: DevicePtr,
    output_tokens: DevicePtr,
) -> Result<()> {
    let mut args = Args::new().u64(state).u64(next_token).u64(output_tokens);
    unsafe {
        launch(
            kernels.advance,
            Launch {
                grid: (1, 1, 1),
                block: (32, 1, 1),
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )
    }
}
