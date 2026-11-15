use anyhow::{ensure, Result};
use half::bf16;
use qwen_ultrafast_cuda::{Buffer, Device, Stream};
use qwen_ultrafast_engine::components;
use qwen_ultrafast_engine::reference;
use qwen_ultrafast_engine::weights::Nvfp4Matrix;
use qwen_ultrafast_kernels::Kernels;
use std::collections::BTreeMap;

fn check(actual: &[bf16], expected: &[bf16], atol: f32, rtol: f32) -> Result<f32> {
    let mut maximum = 0.0f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        ensure!(
            error <= atol + rtol * expected.abs(),
            "mismatch at {index}: {actual} vs {expected}"
        );
    }
    Ok(maximum)
}

fn gdn_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let (key_heads, value_heads, head_dim) = (16, 48, 128);
    let key_dim = key_heads * head_dim;
    let value_dim = value_heads * head_dim;
    let qkv_host = reference::values(2 * key_dim + value_dim, 11);
    let z_host = reference::values(value_dim, 13);
    let a_host = reference::values(value_heads, 17);
    let b_host = reference::values(value_heads, 19);
    let a_log_host: Vec<f32> = (0..value_heads).map(|x| 0.01 * x as f32).collect();
    let dt_bias_host = vec![0.1f32; value_heads];
    let norm_host = reference::values(head_dim, 23);
    let mut state_host: Vec<f32> = reference::values(value_dim * head_dim, 29)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let mut expected_state = state_host.clone();
    let expected = reference::gdn_recurrence(
        &qkv_host,
        &z_host,
        &a_host,
        &b_host,
        &a_log_host,
        &dt_bias_host,
        &norm_host,
        &mut expected_state,
        key_heads,
        value_heads,
        head_dim,
        1e-6,
    );
    let qkv = Buffer::from_host(&qkv_host)?;
    let z = Buffer::from_host(&z_host)?;
    let a = Buffer::from_host(&a_host)?;
    let b = Buffer::from_host(&b_host)?;
    let a_log = Buffer::from_host(&a_log_host)?;
    let dt_bias = Buffer::from_host(&dt_bias_host)?;
    let norm = Buffer::from_host(&norm_host)?;
    let state = Buffer::from_host(&state_host)?;
    let output = Buffer::alloc(value_dim)?;
    let launch = || {
        components::gdn_recurrence(
            kernels,
            stream,
            qkv.ptr(),
            z.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            norm.ptr(),
            state.ptr(),
            output.ptr(),
            0,
            key_heads,
            value_heads,
            head_dim,
            1e-6,
        )
    };
    launch()?;
    stream.synchronize()?;
    let direct = check(&output.download()?, &expected, 0.015, 0.03)?;
    state_host = expected_state;
    let state_error = state
        .download()?
        .iter()
        .zip(state_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    ensure!(state_error < 0.01, "recurrent state error {state_error}");
    stream.begin_capture()?;
    launch()?;
    let graph = stream.end_capture()?.instantiate()?;
    state.upload(
        &reference::values(value_dim * head_dim, 29)
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>(),
    )?;
    graph.launch(stream)?;
    stream.synchronize()?;
    Ok(direct.max(check(&output.download()?, &expected, 0.015, 0.03)?))
}

fn nvfp4_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let (n, k) = (32, 256);
    let input_host = reference::values(k, 31);
    let packed_host: Vec<u8> = (0..n * k / 2).map(|x| (x * 37 % 256) as u8).collect();
    let scale_values = [0x38u8, 0x30, 0x40, 0x34];
    let scales_host: Vec<u8> = (0..n * k / 16).map(|x| scale_values[x % 4]).collect();
    let expected = reference::nvfp4_gemv(&input_host, &packed_host, &scales_host, 0.75, n, k);
    let input = Buffer::from_host(&input_host)?;
    let packed = Buffer::from_host(&packed_host)?;
    let scales = Buffer::from_host(&scales_host)?;
    let global = Buffer::from_host(&[0.75f32])?;
    let output = Buffer::alloc(n)?;
    components::nvfp4_gemv(
        kernels,
        stream,
        input.ptr(),
        Nvfp4Matrix {
            weight: packed.ptr(),
            scales: scales.ptr(),
            global_scale: global.ptr(),
            n,
            k,
        },
        output.ptr(),
    )?;
    stream.synchronize()?;
    check(&output.download()?, &expected, 0.01, 0.02)
}

fn router_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let logits_host = reference::values(512, 37);
    let logits = Buffer::from_host(&logits_host)?;
    let ids = Buffer::<u32>::alloc(10)?;
    let weights = Buffer::<f32>::alloc(10)?;
    components::router(
        kernels,
        stream,
        logits.ptr(),
        ids.ptr(),
        weights.ptr(),
        512,
        10,
    )?;
    stream.synchronize()?;
    let maximum = logits_host
        .iter()
        .map(|x| x.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let probabilities: Vec<f32> = logits_host
        .iter()
        .map(|x| (x.to_f32() - maximum).exp())
        .collect();
    let sum = probabilities.iter().sum::<f32>();
    let mut order: Vec<usize> = (0..512).collect();
    order.sort_by(|a, b| probabilities[*b].total_cmp(&probabilities[*a]));
    let selected_sum = order[..10]
        .iter()
        .map(|x| probabilities[*x] / sum)
        .sum::<f32>();
    let actual_ids = ids.download()?;
    let expected_ids = order[..10].iter().map(|x| *x as u32).collect::<Vec<_>>();
    ensure!(
        actual_ids == expected_ids,
        "router ids differ: {actual_ids:?}"
    );
    let actual_weights = weights.download()?;
    for (slot, id) in order[..10].iter().enumerate() {
        let expected = probabilities[*id] / sum / selected_sum;
        ensure!(
            (actual_weights[slot] - expected).abs() < 1e-5,
            "router weight mismatch"
        );
    }
    Ok(0.0)
}

fn gdn_conv_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let channels = 256;
    let input_host = reference::values(channels, 41);
    let weight_host = reference::values(channels * 4, 43);
    let mut expected_state = reference::values(channels * 3, 47);
    let expected = reference::gdn_conv(&input_host, &weight_host, &mut expected_state);
    let input = Buffer::from_host(&input_host)?;
    let weight = Buffer::from_host(&weight_host)?;
    let state = Buffer::from_host(&reference::values(channels * 3, 47))?;
    let output = Buffer::alloc(channels)?;
    components::gdn_conv(
        kernels,
        stream,
        input.ptr(),
        weight.ptr(),
        state.ptr(),
        output.ptr(),
        channels,
    )?;
    stream.synchronize()?;
    let error = check(&output.download()?, &expected, 0.005, 0.02)?;
    let state_error = state
        .download()?
        .iter()
        .zip(expected_state.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0.0f32, f32::max);
    ensure!(state_error == 0.0, "convolution state mismatch");
    Ok(error)
}

fn qsa_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let (query_heads, kv_heads, head_dim) = (2, 1, 256);
    let query_dim = query_heads * head_dim;
    let projected_query_host = reference::values(2 * query_dim, 53);
    let key_host = reference::values(head_dim, 59);
    let value_host = reference::values(head_dim, 61);
    let query_norm = Buffer::from_host(&reference::values(head_dim, 67))?;
    let key_norm = Buffer::from_host(&reference::values(head_dim, 71))?;
    let projected_query = Buffer::from_host(&projected_query_host)?;
    let projected_key = Buffer::from_host(&key_host)?;
    let projected_value = Buffer::from_host(&value_host)?;
    let query = Buffer::<bf16>::alloc(query_dim)?;
    let gate = Buffer::<bf16>::alloc(query_dim)?;
    let key_cache = Buffer::<bf16>::alloc(head_dim)?;
    let value_cache = Buffer::<bf16>::alloc(head_dim)?;
    let cosines = Buffer::from_host(&[1.0f32; 32])?;
    let sines = Buffer::from_host(&[0.0f32; 32])?;
    let state = Buffer::from_host(&[qwen_ultrafast_engine::ops::State::default()])?;
    let partial_values = Buffer::<f32>::alloc(16 * query_dim)?;
    let partial_max = Buffer::<f32>::alloc(16 * query_heads)?;
    let partial_sum = Buffer::<f32>::alloc(16 * query_heads)?;
    let output = Buffer::alloc(query_dim)?;
    components::qsa_prepare(
        kernels,
        stream,
        projected_query.ptr(),
        projected_key.ptr(),
        projected_value.ptr(),
        query_norm.ptr(),
        key_norm.ptr(),
        query.ptr(),
        gate.ptr(),
        key_cache.ptr(),
        value_cache.ptr(),
        state.ptr(),
        cosines.ptr(),
        sines.ptr(),
        query_heads,
        kv_heads,
        head_dim,
        64,
        1,
        1e-6,
    )?;
    components::qsa_attention(
        kernels,
        stream,
        query.ptr(),
        key_cache.ptr(),
        value_cache.ptr(),
        state.ptr(),
        partial_values.ptr(),
        partial_max.ptr(),
        partial_sum.ptr(),
        gate.ptr(),
        output.ptr(),
        query_heads,
        kv_heads,
        head_dim,
        1,
        16,
        1.0 / 16.0,
    )?;
    stream.synchronize()?;
    let expected: Vec<bf16> = (0..query_dim)
        .map(|i| {
            let head = i / head_dim;
            let d = i % head_dim;
            let gate = projected_query_host[head * 2 * head_dim + head_dim + d].to_f32();
            bf16::from_f32(value_host[d].to_f32() / (1.0 + (-gate).exp()))
        })
        .collect();
    check(&output.download()?, &expected, 0.01, 0.02)
}

fn ple_test(kernels: &Kernels, stream: &Stream) -> Result<f32> {
    let eos = 248044u32;
    let multipliers = [0x4000_0000_0000_0001u64, 5, 7];
    let vocabulary_sizes = [
        101u64, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
    ];
    let mut offsets = [0u64; 16];
    for i in 1..16 {
        offsets[i] = offsets[i - 1] + vocabulary_sizes[i - 1];
    }
    let history = Buffer::from_host(&[eos, eos])?;
    let rows = Buffer::<u64>::alloc(16)?;
    let multipliers_gpu = Buffer::from_host(&multipliers)?;
    let vocabulary_sizes_gpu = Buffer::from_host(&vocabulary_sizes)?;
    let offsets_gpu = Buffer::from_host(&offsets)?;
    let launch_hash = |token| {
        components::ple_hash(
            kernels,
            stream,
            token,
            history.ptr(),
            multipliers_gpu.ptr(),
            vocabulary_sizes_gpu.ptr(),
            offsets_gpu.ptr(),
            rows.ptr(),
            eos,
        )
    };
    launch_hash(42)?;
    stream.synchronize()?;
    let actual = rows.download()?;
    for head in 0..16 {
        let mut mixed =
            42u64.wrapping_mul(multipliers[0]) ^ (eos as u64).wrapping_mul(multipliers[1]);
        if head >= 8 {
            mixed ^= (eos as u64).wrapping_mul(multipliers[2]);
        }
        let signed_mixed = mixed as i64;
        ensure!(
            signed_mixed < 0,
            "PLE test must cover negative signed hashes"
        );
        let expected =
            signed_mixed.rem_euclid(vocabulary_sizes[head] as i64) as u64 + offsets[head];
        ensure!(actual[head] == expected, "PLE hash mismatch");
    }
    launch_hash(43)?;
    stream.synchronize()?;
    let actual_history = history.download()?;
    ensure!(
        actual_history == [43, 42],
        "PLE token history mismatch: {actual_history:?}"
    );

    let channels = 256;
    let gated_host = reference::values(channels, 73);
    let normed_host = reference::values(channels, 79);
    let weight_host = reference::values(channels * 4, 83);
    let history_host = reference::values(channels * 9, 89);
    let expected: Vec<bf16> = (0..channels)
        .map(|channel| {
            let mut value = weight_host[channel * 4 + 3].to_f32() * normed_host[channel].to_f32();
            value += weight_host[channel * 4].to_f32() * history_host[channel * 9].to_f32();
            value += weight_host[channel * 4 + 1].to_f32() * history_host[channel * 9 + 3].to_f32();
            value += weight_host[channel * 4 + 2].to_f32() * history_host[channel * 9 + 6].to_f32();
            bf16::from_f32(gated_host[channel].to_f32() + value / (1.0 + (-value).exp()))
        })
        .collect();
    let output = Buffer::alloc(channels)?;
    let gated = Buffer::from_host(&gated_host)?;
    let normed = Buffer::from_host(&normed_host)?;
    let weight = Buffer::from_host(&weight_host)?;
    let convolution_history = Buffer::from_host(&history_host)?;
    components::ple_conv(
        kernels,
        stream,
        gated.ptr(),
        normed.ptr(),
        weight.ptr(),
        convolution_history.ptr(),
        output.ptr(),
        channels,
    )?;
    stream.synchronize()?;
    check(&output.download()?, &expected, 0.005, 0.02)
}

fn main() -> Result<()> {
    let ordinal = std::env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix("--device=").and_then(|x| x.parse().ok()))
        .unwrap_or(0);
    let _device = Device::new(ordinal)?;
    let stream = Stream::new()?;
    let kernels = Kernels::load()?;
    let mut errors = BTreeMap::new();
    errors.insert("gdn_recurrence", gdn_test(&kernels, &stream)?);
    errors.insert("nvfp4_gemv", nvfp4_test(&kernels, &stream)?);
    errors.insert("router_top10", router_test(&kernels, &stream)?);
    errors.insert("gdn_convolution", gdn_conv_test(&kernels, &stream)?);
    errors.insert("qsa_single_token", qsa_test(&kernels, &stream)?);
    errors.insert("ple_hash_and_convolution", ple_test(&kernels, &stream)?);
    println!("{}", serde_json::to_string_pretty(&errors)?);
    Ok(())
}
