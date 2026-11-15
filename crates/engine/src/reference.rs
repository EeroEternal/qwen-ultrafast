use half::bf16;

pub fn values(len: usize, seed: u64) -> Vec<bf16> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            bf16::from_f32((unit - 0.5) * 0.5)
        })
        .collect()
}

pub fn gdn_conv(input: &[bf16], weight: &[bf16], state: &mut [bf16]) -> Vec<bf16> {
    let mut output = vec![bf16::ZERO; input.len()];
    for channel in 0..input.len() {
        let mut values = [0.0f32; 4];
        for (index, value) in values.iter_mut().enumerate().take(3) {
            *value = state[channel * 3 + index].to_f32();
        }
        values[3] = input[channel].to_f32();
        let mut sum = 0.0f32;
        for index in 0..4 {
            sum += values[index] * weight[channel * 4 + index].to_f32();
        }
        output[channel] = bf16::from_f32(sum / (1.0 + (-sum).exp()));
        state[channel * 3] = state[channel * 3 + 1];
        state[channel * 3 + 1] = state[channel * 3 + 2];
        state[channel * 3 + 2] = input[channel];
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence(
    qkv: &[bf16],
    z: &[bf16],
    a: &[bf16],
    b: &[bf16],
    a_log: &[f32],
    dt_bias: &[f32],
    norm: &[bf16],
    state: &mut [f32],
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Vec<bf16> {
    let key_dim = key_heads * head_dim;
    let mut output = vec![bf16::ZERO; value_heads * head_dim];
    for head in 0..value_heads {
        let key_head = head / (value_heads / key_heads);
        let mut query: Vec<f32> = (0..head_dim)
            .map(|d| qkv[key_head * head_dim + d].to_f32())
            .collect();
        let mut key: Vec<f32> = (0..head_dim)
            .map(|d| qkv[key_dim + key_head * head_dim + d].to_f32())
            .collect();
        let q_norm = (query.iter().map(|x| x * x).sum::<f32>() + epsilon)
            .sqrt()
            .recip();
        let k_norm = (key.iter().map(|x| x * x).sum::<f32>() + epsilon)
            .sqrt()
            .recip();
        for d in 0..head_dim {
            query[d] *= q_norm / (head_dim as f32).sqrt();
            key[d] *= k_norm;
        }
        let decay = (-a_log[head].exp() * (a[head].to_f32() + dt_bias[head]).exp().ln_1p()).exp();
        let beta = 1.0 / (1.0 + (-b[head].to_f32()).exp());
        let cells = &mut state[head * head_dim * head_dim..(head + 1) * head_dim * head_dim];
        for v in 0..head_dim {
            let value = qkv[2 * key_dim + head * head_dim + v].to_f32();
            let mut memory = 0.0f32;
            for k in 0..head_dim {
                cells[k * head_dim + v] *= decay;
                memory += cells[k * head_dim + v] * key[k];
            }
            let delta = (value - memory) * beta;
            let mut result = 0.0f32;
            for k in 0..head_dim {
                cells[k * head_dim + v] += key[k] * delta;
                result += cells[k * head_dim + v] * query[k];
            }
            output[head * head_dim + v] = bf16::from_f32(result);
        }
        let variance = output[head * head_dim..(head + 1) * head_dim]
            .iter()
            .map(|x| {
                let value = x.to_f32();
                value * value
            })
            .sum::<f32>()
            / head_dim as f32;
        for (v, norm_value) in norm.iter().enumerate().take(head_dim) {
            let index = head * head_dim + v;
            let normalized =
                bf16::from_f32(output[index].to_f32() * (variance + epsilon).sqrt().recip());
            let gate = z[index].to_f32();
            output[index] =
                bf16::from_f32(normalized.to_f32() * norm_value.to_f32() / (1.0 + (-gate).exp()));
        }
    }
    output
}

pub fn nvfp4_gemv(
    input: &[bf16],
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    n: usize,
    k: usize,
) -> Vec<bf16> {
    const FP4: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let fp8 = |raw: u8| {
        let sign = if raw & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = ((raw >> 3) & 15) as i32;
        let mantissa = (raw & 7) as f32;
        if exponent == 0 {
            sign * mantissa * 2.0f32.powi(-9)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    let mut output = vec![bf16::ZERO; n];
    for column in 0..n {
        let mut sum = 0.0f32;
        for group in 0..k / 16 {
            let mut dot = 0.0f32;
            for j in 0..16 {
                let byte = packed[column * (k / 2) + group * 8 + j / 2];
                let code = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
                dot += input[group * 16 + j].to_f32() * FP4[code as usize];
            }
            sum += fp8(scales[column * (k / 16) + group]) * dot;
        }
        output[column] = bf16::from_f32(sum * global_scale);
    }
    output
}

pub fn nvfp4_quantize(input: &[bf16], input_scale_inv: f32) -> (Vec<u8>, Vec<u8>) {
    const FP4: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let fp8 = |raw: u8| {
        let exponent = ((raw >> 3) & 15) as i32;
        let mantissa = (raw & 7) as f32;
        if exponent == 0 {
            mantissa * 2.0f32.powi(-9)
        } else {
            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    let nearest = |value: f32, values: &[f32]| {
        let mut best = 0;
        let mut error = f32::INFINITY;
        for (index, candidate) in values.iter().enumerate() {
            let candidate_error = (value - candidate).abs();
            if candidate_error < error
                || (candidate_error == error && best % 2 == 1 && index % 2 == 0)
            {
                best = index;
                error = candidate_error;
            }
        }
        best
    };
    let fp8_values: Vec<f32> = (0..=0x7e).map(fp8).collect();
    let mut packed = Vec::with_capacity(input.len() / 2);
    let mut scales = Vec::with_capacity(input.len() / 16);
    for group in 0..input.len() / 16 {
        let values = &input[group * 16..(group + 1) * 16];
        let maximum = values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0f32, f32::max);
        let raw_scale = nearest(maximum * input_scale_inv / 6.0, &fp8_values) as u8;
        scales.push(raw_scale);
        let inverse = if raw_scale == 0 {
            0.0
        } else {
            input_scale_inv / fp8(raw_scale)
        };
        for pair in 0..8 {
            let low = values[2 * pair].to_f32() * inverse;
            let high = values[2 * pair + 1].to_f32() * inverse;
            let mut low = nearest(low, &FP4);
            let mut high = nearest(high, &FP4);
            if low == 0 && values[2 * pair].to_f32().is_sign_negative() {
                low = 8;
            }
            if high == 0 && values[2 * pair + 1].to_f32().is_sign_negative() {
                high = 8;
            }
            packed.push((low | (high << 4)) as u8);
        }
    }
    (packed, scales)
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv_w4a4(
    input: &[bf16],
    input_scale: f32,
    input_scale_inv: f32,
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    n: usize,
    k: usize,
) -> Vec<bf16> {
    const FP4: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let fp8 = |raw: u8| {
        let exponent = ((raw >> 3) & 15) as i32;
        let mantissa = (raw & 7) as f32;
        if exponent == 0 {
            mantissa * 2.0f32.powi(-9)
        } else {
            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    };
    let (input_packed, input_scales) = nvfp4_quantize(input, input_scale_inv);
    let mut output = vec![bf16::ZERO; n];
    for column in 0..n {
        let mut sum = 0.0f32;
        for group in 0..k / 16 {
            let mut dot = 0.0f32;
            for j in 0..16 {
                let input_byte = input_packed[group * 8 + j / 2];
                let weight_byte = packed[column * (k / 2) + group * 8 + j / 2];
                let input_code = if j % 2 == 0 {
                    input_byte & 15
                } else {
                    input_byte >> 4
                };
                let weight_code = if j % 2 == 0 {
                    weight_byte & 15
                } else {
                    weight_byte >> 4
                };
                dot += FP4[input_code as usize] * FP4[weight_code as usize];
            }
            sum += fp8(input_scales[group]) * fp8(scales[column * (k / 16) + group]) * dot;
        }
        output[column] = bf16::from_f32(sum * global_scale * input_scale);
    }
    output
}
