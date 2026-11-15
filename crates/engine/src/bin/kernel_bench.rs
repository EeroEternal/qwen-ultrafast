use anyhow::{bail, ensure, Context, Result};
use half::bf16;
use qwen_ultrafast_cuda::{Buffer, Device, Event, Stream};
use qwen_ultrafast_engine::{ops, reference, weights::Matrix};
use qwen_ultrafast_kernels::Kernels;
use serde_json::json;

struct Options {
    threads: u32,
    baseline_threads: u32,
    rounds: usize,
    iterations: usize,
    n: usize,
    k: usize,
    device: i32,
}

fn options() -> Result<Options> {
    let mut options = Options {
        threads: 128,
        baseline_threads: 128,
        rounds: 19,
        iterations: 128,
        n: 2560,
        k: 6144,
        device: 0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = || {
            args.get(index + 1)
                .cloned()
                .context(format!("{} needs a value", args[index]))
        };
        match args[index].as_str() {
            "--threads" => options.threads = value()?.parse()?,
            "--baseline-threads" => options.baseline_threads = value()?.parse()?,
            "--rounds" => options.rounds = value()?.parse()?,
            "--iterations" => options.iterations = value()?.parse()?,
            "--n" => options.n = value()?.parse()?,
            "--k" => options.k = value()?.parse()?,
            "--device" => options.device = value()?.parse()?,
            flag => bail!("unknown option {flag}"),
        }
        index += 2;
    }
    ensure!(matches!(options.threads, 64 | 128 | 256 | 512));
    ensure!(matches!(options.baseline_threads, 64 | 128 | 256 | 512));
    ensure!((5..=101).contains(&options.rounds));
    ensure!((1..=1000).contains(&options.iterations));
    ensure!(options.n > 0 && options.k > 0 && options.k % 8 == 0);
    Ok(options)
}

fn error(actual: &[bf16], expected: &[bf16]) -> Result<f32> {
    let mut maximum = 0.0f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual.to_f32() - expected.to_f32()).abs();
        ensure!(
            difference <= 0.01 + 0.02 * expected.to_f32().abs(),
            "mismatch at {index}: {} vs {}",
            actual.to_f32(),
            expected.to_f32()
        );
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}

fn measure(
    kernels: &Kernels,
    stream: &Stream,
    input: &Buffer<bf16>,
    weight: Matrix,
    output: &Buffer<bf16>,
    threads: u32,
    iterations: usize,
) -> Result<f64> {
    let start = Event::timed()?;
    let stop = Event::timed()?;
    start.record(stream)?;
    for _ in 0..iterations {
        ops::gemv_with_threads(
            kernels,
            stream,
            input.ptr(),
            weight,
            output.ptr(),
            false,
            threads,
        )?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(stop.elapsed_ms(&start)? as f64 * 1000.0 / iterations as f64)
}

fn main() -> Result<()> {
    let options = options()?;
    let _device = Device::new(options.device)?;
    let stream = Stream::new()?;
    let kernels = Kernels::load()?;
    let input_host = reference::values(options.k, 31);
    let weight_host = reference::values(options.n * options.k, 43);
    let mut expected = vec![bf16::ZERO; options.n];
    for (row, output) in expected.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for column in 0..options.k {
            sum += input_host[column].to_f32() * weight_host[row * options.k + column].to_f32();
        }
        *output = bf16::from_f32(sum);
    }
    let input = Buffer::from_host(&input_host)?;
    let weight = Buffer::from_host(&weight_host)?;
    let matrix = Matrix {
        ptr: weight.ptr(),
        n: options.n,
        k: options.k,
    };
    let baseline = Buffer::<bf16>::alloc(options.n)?;
    let candidate = Buffer::<bf16>::alloc(options.n)?;
    for _ in 0..8 {
        ops::gemv_with_threads(
            &kernels,
            &stream,
            input.ptr(),
            matrix,
            baseline.ptr(),
            false,
            options.baseline_threads,
        )?;
        ops::gemv_with_threads(
            &kernels,
            &stream,
            input.ptr(),
            matrix,
            candidate.ptr(),
            false,
            options.threads,
        )?;
    }
    stream.synchronize()?;
    let baseline_error = error(&baseline.download()?, &expected)?;
    let candidate_error = error(&candidate.download()?, &expected)?;
    let mut baseline_us = Vec::with_capacity(options.rounds);
    let mut candidate_us = Vec::with_capacity(options.rounds);
    for round in 0..options.rounds {
        if round % 2 == 0 {
            baseline_us.push(measure(
                &kernels,
                &stream,
                &input,
                matrix,
                &baseline,
                options.baseline_threads,
                options.iterations,
            )?);
            candidate_us.push(measure(
                &kernels,
                &stream,
                &input,
                matrix,
                &candidate,
                options.threads,
                options.iterations,
            )?);
        } else {
            candidate_us.push(measure(
                &kernels,
                &stream,
                &input,
                matrix,
                &candidate,
                options.threads,
                options.iterations,
            )?);
            baseline_us.push(measure(
                &kernels,
                &stream,
                &input,
                matrix,
                &baseline,
                options.baseline_threads,
                options.iterations,
            )?);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "target": "bf16-gemv",
            "shape": {"n": options.n, "k": options.k},
            "config": {"threads": options.threads},
            "baseline_config": {"threads": options.baseline_threads},
            "correct": true,
            "baseline_max_error": baseline_error,
            "candidate_max_error": candidate_error,
            "baseline_us": baseline_us,
            "candidate_us": candidate_us,
            "timer": "CUDA events around repeated launches; alternating AB/BA order",
        }))?
    );
    Ok(())
}
