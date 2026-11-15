use anyhow::{bail, Context, Result};
use qwen_ultrafast_cuda::{can_access_peer, Device};
use qwen_ultrafast_engine::{components, runtime::Runtime};
use qwen_ultrafast_model::{Checkpoint, Config};
use qwen_ultrafast_nccl as nccl;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Barrier, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

struct Options {
    model: PathBuf,
    prompt: String,
    decode: usize,
    gpus: Vec<i32>,
    max_sequence: usize,
    load_only: bool,
    graph: bool,
}

fn options() -> Result<Options> {
    let mut options = Options {
        model: PathBuf::from("models/Qwen3.8-Flash-Next-NVFP4"),
        prompt: "Hello".into(),
        decode: 1,
        gpus: (0..8).collect(),
        max_sequence: 2048,
        load_only: false,
        graph: false,
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
            "--model" => options.model = value()?.into(),
            "--prompt" => options.prompt = value()?,
            "--decode" => options.decode = value()?.parse()?,
            "--gpus" => {
                options.gpus = value()?
                    .split(',')
                    .map(|part| part.parse().map_err(Into::into))
                    .collect::<Result<_>>()?
            }
            "--max-seq" => options.max_sequence = value()?.parse()?,
            "--load-only" => {
                options.load_only = true;
                index += 1;
                continue;
            }
            "--graph" => {
                options.graph = true;
                index += 1;
                continue;
            }
            flag => bail!("unknown option {flag}"),
        }
        index += 2;
    }
    if options.decode == 0 || options.gpus.is_empty() {
        bail!("decode and GPU count must be positive");
    }
    Ok(options)
}

fn main() -> Result<()> {
    let options = options()?;
    let config = Config::load(&options.model)?;
    let checkpoint = Checkpoint::open(&options.model)?;
    let tokenizer = Tokenizer::from_file(options.model.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    let prompt = tokenizer
        .encode(options.prompt.clone(), true)
        .map_err(|error| anyhow::anyhow!("tokenize: {error}"))?;
    let prompt_ids: Vec<u32> = prompt.get_ids().to_vec();
    if prompt_ids.is_empty() || prompt_ids.len() + options.decode > options.max_sequence {
        bail!("prompt/decode exceeds sequence capacity");
    }
    let world_size = options.gpus.len();
    let context_devices = options
        .gpus
        .iter()
        .map(|ordinal| Device::new(*ordinal))
        .collect::<Result<Vec<_>>>()?;
    let contexts: Vec<usize> = context_devices
        .iter()
        .map(|device| device.context() as usize)
        .collect();
    let gpu_reduction = world_size > 1
        && options.gpus.iter().all(|left| {
            options
                .gpus
                .iter()
                .all(|right| left == right || can_access_peer(*left, *right).unwrap_or(false))
        });
    if std::env::var_os("NCCL_ALGO").is_none() {
        std::env::set_var("NCCL_ALGO", "TREE");
    }
    let nccl_id = if world_size > 1 && std::env::var_os("QWEN_NCCL_LIBRARY").is_some() {
        Some(nccl::Comm::unique_id()?)
    } else {
        None
    };
    let peer_buffers = Mutex::new(vec![0u64; world_size]);
    let barrier = Barrier::new(world_size);
    let reductions = Mutex::new(vec![Vec::new(); world_size]);
    let load_failures = Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for (rank, ordinal) in options.gpus.iter().copied().enumerate() {
            let config = config.clone();
            let checkpoint = &checkpoint;
            let barrier = &barrier;
            let reductions = &reductions;
            let load_failures = &load_failures;
            let peer_buffers = &peer_buffers;
            let contexts = &contexts;
            let prompt_ids = &prompt_ids;
            let options = &options;
            workers.push(scope.spawn(move || -> Result<(Vec<u32>, f64, f64, f64)> {
                let mut runtime = match Runtime::new(
                    ordinal,
                    rank,
                    world_size,
                    checkpoint,
                    config,
                    options.max_sequence,
                ) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        load_failures
                            .lock()
                            .unwrap()
                            .push(format!("rank {rank}: {error:#}"));
                        barrier.wait();
                        return Err(error);
                    }
                };
                barrier.wait();
                let failures = load_failures.lock().unwrap();
                if !failures.is_empty() {
                    bail!("model loading failed: {}", failures.join("; "));
                }
                drop(failures);
                let comm = match nccl_id {
                    Some(id) => Some(nccl::Comm::new(world_size, rank, id)?),
                    None => None,
                };
                if gpu_reduction {
                    runtime.enable_peers(contexts)?;
                    peer_buffers.lock().unwrap()[rank] = runtime.routed_ptr();
                    barrier.wait();
                }
                if rank == 0 {
                    eprintln!(
                        "loaded {:.2} GiB on GPU {} in {:.1}s",
                        runtime.weights.bytes as f64 / (1u64 << 30) as f64,
                        ordinal,
                        started.elapsed().as_secs_f32()
                    );
                }
                if options.load_only {
                    return Ok((Vec::new(), 0.0, 0.0, 0.0));
                }
                let inference_started = Instant::now();
                let mut generated = Vec::new();
                for (position, token) in prompt_ids.iter().copied().enumerate() {
                    runtime.forward_token(
                        token,
                        position,
                        position,
                        &mut |kernels, stream, routed| {
                            if let Some(comm) = &comm {
                                comm.all_reduce(stream, routed)?;
                            } else if gpu_reduction {
                                stream.synchronize()?;
                                barrier.wait();
                                let mut peers = [0u64; 8];
                                peers[..world_size].copy_from_slice(&peer_buffers.lock().unwrap());
                                components::peer_sum(
                                    kernels,
                                    stream,
                                    peers,
                                    world_size,
                                    routed.ptr(),
                                    routed.len(),
                                )?;
                                stream.synchronize()?;
                                barrier.wait();
                            } else {
                                let values = routed.download()?;
                                reductions.lock().unwrap()[rank] = values;
                                barrier.wait();
                                let slots = reductions.lock().unwrap();
                                let mut total = vec![0.0f32; slots[rank].len()];
                                for slot in slots.iter() {
                                    for (output, value) in total.iter_mut().zip(slot) {
                                        *output += value;
                                    }
                                }
                                drop(slots);
                                routed.upload(&total)?;
                                barrier.wait();
                            }
                            Ok(())
                        },
                    )?;
                }
                let mut current = runtime.output_tokens.download()?[prompt_ids.len() - 1];
                generated.push(current);
                let prefill_seconds = inference_started.elapsed().as_secs_f64();
                let capture_started = Instant::now();
                let graph = if options.graph {
                    let comm = comm.as_ref().context("--graph requires NCCL")?;
                    Some(runtime.capture_nccl(comm)?)
                } else {
                    None
                };
                let capture_seconds = capture_started.elapsed().as_secs_f64();
                let decode_started = Instant::now();
                for step in 1..options.decode {
                    let position = prompt_ids.len() + step - 1;
                    if let Some(graph) = &graph {
                        runtime.prepare_input(current, position, position)?;
                        runtime.launch_graph(graph)?;
                    } else {
                        runtime.forward_token(
                            current,
                            position,
                            position,
                            &mut |kernels, stream, routed| {
                                if let Some(comm) = &comm {
                                    comm.all_reduce(stream, routed)?;
                                } else if gpu_reduction {
                                    stream.synchronize()?;
                                    barrier.wait();
                                    let mut peers = [0u64; 8];
                                    peers[..world_size]
                                        .copy_from_slice(&peer_buffers.lock().unwrap());
                                    components::peer_sum(
                                        kernels,
                                        stream,
                                        peers,
                                        world_size,
                                        routed.ptr(),
                                        routed.len(),
                                    )?;
                                    stream.synchronize()?;
                                    barrier.wait();
                                } else {
                                    let values = routed.download()?;
                                    reductions.lock().unwrap()[rank] = values;
                                    barrier.wait();
                                    let slots = reductions.lock().unwrap();
                                    let mut total = vec![0.0f32; slots[rank].len()];
                                    for slot in slots.iter() {
                                        for (output, value) in total.iter_mut().zip(slot) {
                                            *output += value;
                                        }
                                    }
                                    drop(slots);
                                    routed.upload(&total)?;
                                    barrier.wait();
                                }
                                Ok(())
                            },
                        )?;
                    }
                    current = runtime.output_tokens.download()?[position];
                    generated.push(current);
                }
                Ok((
                    generated,
                    prefill_seconds,
                    capture_seconds,
                    decode_started.elapsed().as_secs_f64(),
                ))
            }));
        }
        let mut generated = None;
        for (rank, worker) in workers.into_iter().enumerate() {
            let result = worker
                .join()
                .map_err(|_| anyhow::anyhow!("rank {rank} panicked"))??;
            if rank == 0 {
                generated = Some(result);
            }
        }
        let (generated, prefill_seconds, capture_seconds, decode_seconds) =
            generated.context("no rank zero")?;
        if !options.load_only {
            let text = tokenizer
                .decode(&generated, false)
                .map_err(|error| anyhow::anyhow!("decode tokens: {error}"))?;
            let decode_tokens_per_second = if generated.len() > 1 && decode_seconds > 0.0 {
                (generated.len() - 1) as f64 / decode_seconds
            } else {
                0.0
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "prompt_tokens": prompt_ids,
                    "tokens": generated,
                    "text": text,
                    "elapsed_seconds": started.elapsed().as_secs_f64(),
                    "prefill_seconds": prefill_seconds,
                    "graph_capture_seconds": capture_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": decode_tokens_per_second,
                    "mode": if options.graph { "correctness_nccl_cuda_graph" } else if nccl_id.is_some() { "correctness_nccl_reduction" } else if gpu_reduction { "correctness_gpu_peer_reduction" } else { "correctness_host_reduction" },
                }))?
            );
        }
        Ok(())
    })
}
