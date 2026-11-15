use anyhow::Result;
use half::bf16;
use qwen_ultrafast_cuda::{Buffer, Device, Stream};
use qwen_ultrafast_nccl as nccl;
use serde_json::json;
use std::sync::Barrier;
use std::time::Instant;

fn main() -> Result<()> {
    let world_size = 8;
    let id = nccl::Comm::unique_id()?;
    let barrier = Barrier::new(world_size);
    let bf16_mode = std::env::var_os("QWEN_NCCL_BF16").is_some();
    let iterations = std::env::var("QWEN_NCCL_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for rank in 0..world_size {
            let barrier = &barrier;
            workers.push(scope.spawn(move || -> Result<f64> {
                let _device = Device::new(rank as i32)?;
                let stream = Stream::new()?;
                let comm = nccl::Comm::new(world_size, rank, id)?;
                let f32_buffer = (!bf16_mode)
                    .then(|| Buffer::from_host(&vec![rank as f32 + 1.0; 2560]))
                    .transpose()?;
                let bf16_buffer = bf16_mode
                    .then(|| Buffer::from_host(&vec![bf16::from_f32(rank as f32 + 1.0); 2560]))
                    .transpose()?;
                let graph = if let Some(buffer) = &bf16_buffer {
                    stream.begin_capture()?;
                    comm.all_reduce_bf16(&stream, buffer)?;
                    stream.end_capture()?.instantiate()?
                } else {
                    let buffer = f32_buffer.as_ref().unwrap();
                    stream.begin_capture()?;
                    comm.all_reduce(&stream, buffer)?;
                    stream.end_capture()?.instantiate()?
                };
                barrier.wait();
                graph.launch(&stream)?;
                stream.synchronize()?;
                if let Some(buffer) = &bf16_buffer {
                    let values = buffer.download()?;
                    assert!(values.iter().all(|value| value.to_f32() == 36.0));
                } else {
                    let values = f32_buffer.as_ref().unwrap().download()?;
                    assert!(values.iter().all(|value| *value == 36.0));
                }
                for _ in 1..10 {
                    graph.launch(&stream)?;
                }
                stream.synchronize()?;
                let started = Instant::now();
                for _ in 0..iterations {
                    graph.launch(&stream)?;
                }
                stream.synchronize()?;
                Ok(started.elapsed().as_secs_f64())
            }));
        }
        let mut elapsed: f64 = 0.0;
        for worker in workers {
            elapsed = elapsed.max(
                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("worker panicked"))??,
            );
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "mode": if bf16_mode { "bf16" } else { "fp32" },
                "algorithm": std::env::var("NCCL_ALGO").unwrap_or_else(|_| "default".into()),
                "protocol": std::env::var("NCCL_PROTO").unwrap_or_else(|_| "default".into()),
                "iterations": iterations,
                "elapsed_seconds": elapsed,
                "microseconds_per_allreduce": elapsed * 1e6 / iterations as f64,
            }))?
        );
        Ok(())
    })
}
