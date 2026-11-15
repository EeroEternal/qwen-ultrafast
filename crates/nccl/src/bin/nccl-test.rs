use anyhow::{ensure, Result};
use qwen_ultrafast_cuda::{Buffer, Device, Stream};
use qwen_ultrafast_nccl::Comm;

fn main() -> Result<()> {
    let id = Comm::unique_id()?;
    let results = std::thread::scope(|scope| -> Result<Vec<_>> {
        let mut workers = Vec::new();
        for rank in 0..2 {
            workers.push(scope.spawn(move || -> Result<Vec<f32>> {
                let _device = Device::new(rank as i32)?;
                let stream = Stream::new()?;
                let comm = Comm::new(2, rank, id)?;
                let buffer = Buffer::from_host(&[rank as f32 + 1.0, 2.0, 3.0, 4.0])?;
                comm.all_reduce(&stream, &buffer)?;
                stream.synchronize()?;
                buffer.download()
            }));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    })?;
    for values in results {
        ensure!(
            values == [3.0, 4.0, 6.0, 8.0],
            "NCCL result mismatch: {values:?}"
        );
        println!("{values:?}");
    }
    Ok(())
}
