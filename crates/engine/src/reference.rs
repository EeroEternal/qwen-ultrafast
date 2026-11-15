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
