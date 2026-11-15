use anyhow::{bail, ensure, Context, Result};
use half::bf16;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    file: usize,
    start: usize,
    end: usize,
}

pub struct Checkpoint {
    maps: Vec<Mmap>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl Checkpoint {
    pub fn open(directory: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(directory)?
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .collect();
        files.sort();
        ensure!(
            !files.is_empty(),
            "no safetensors files in {}",
            directory.display()
        );
        let mut maps = Vec::new();
        let mut tensors = HashMap::new();
        for (file_index, path) in files.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let map = unsafe { Mmap::map(&file)? };
            ensure!(map.len() >= 8, "{} is not safetensors", path.display());
            let header_size = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            ensure!(8 + header_size <= map.len(), "invalid safetensors header");
            let header: Value = serde_json::from_slice(&map[8..8 + header_size])?;
            for (name, value) in header.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = value["data_offsets"].as_array().unwrap();
                let start = 8 + header_size + offsets[0].as_u64().unwrap() as usize;
                let end = 8 + header_size + offsets[1].as_u64().unwrap() as usize;
                ensure!(
                    start <= end && end <= map.len(),
                    "invalid tensor range for {name}"
                );
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype: value["dtype"].as_str().unwrap().to_string(),
                        shape: value["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|dimension| dimension.as_u64().unwrap() as usize)
                            .collect(),
                        file: file_index,
                        start,
                        end,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Self { maps, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self.info(name)?;
        Ok(&self.maps[tensor.file][tensor.start..tensor.end])
    }

    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
        let tensor = self.info(name)?;
        if tensor.dtype != "BF16" || tensor.shape != shape {
            bail!(
                "{name}: expected BF16{shape:?}, found {}{:?}",
                tensor.dtype,
                tensor.shape
            );
        }
        let bytes = self.bytes(name)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    pub fn fp8_row(
        &self,
        prefix: &str,
        rows: usize,
        row: usize,
        width: usize,
        global_scale: f32,
    ) -> Result<Vec<bf16>> {
        ensure!(row < rows, "FP8 row is out of range");
        let weight_name = format!("{prefix}.weight");
        let tensor = self.info(&weight_name)?;
        ensure!(
            tensor.dtype == "F8_E4M3" && tensor.shape == [rows, width],
            "FP8 row shape mismatch"
        );
        let bytes = &self.bytes(&weight_name)?[row * width..(row + 1) * width];
        Ok(bytes
            .iter()
            .map(|raw| {
                let sign = if raw & 0x80 == 0 { 1.0 } else { -1.0 };
                let exponent = ((raw >> 3) & 15) as i32;
                let mantissa = (raw & 7) as f32;
                let value = if exponent == 0 {
                    sign * mantissa * 2.0f32.powi(-9)
                } else {
                    sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
                };
                bf16::from_f32(value * global_scale)
            })
            .collect())
    }

    pub fn nvfp4_row(
        &self,
        prefix: &str,
        rows: usize,
        row: usize,
        width: usize,
        global_scale: f32,
    ) -> Result<Vec<bf16>> {
        const FP4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        ensure!(row < rows && width % 16 == 0, "invalid NVFP4 row");
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        ensure!(
            self.info(&weight_name)?.shape == [rows, width / 2],
            "NVFP4 weight shape mismatch"
        );
        ensure!(
            self.info(&scale_name)?.shape == [rows, width / 16],
            "NVFP4 scale shape mismatch"
        );
        let packed = &self.bytes(&weight_name)?[row * width / 2..(row + 1) * width / 2];
        let scales = &self.bytes(&scale_name)?[row * width / 16..(row + 1) * width / 16];
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
        let mut output = Vec::with_capacity(width);
        for group in 0..width / 16 {
            for j in 0..16 {
                let byte = packed[group * 8 + j / 2];
                let code = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
                output.push(bf16::from_f32(
                    FP4[code as usize] * fp8(scales[group]) * global_scale,
                ));
            }
        }
        Ok(output)
    }

    pub fn u8(&self, name: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let tensor = self.info(name)?;
        if tensor.dtype != "U8" || tensor.shape != shape {
            bail!(
                "{name}: expected U8{shape:?}, found {}{:?}",
                tensor.dtype,
                tensor.shape
            );
        }
        Ok(self.bytes(name)?.to_vec())
    }

    pub fn fp8(&self, name: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let tensor = self.info(name)?;
        if tensor.dtype != "F8_E4M3" || tensor.shape != shape {
            bail!(
                "{name}: expected F8_E4M3{shape:?}, found {}{:?}",
                tensor.dtype,
                tensor.shape
            );
        }
        Ok(self.bytes(name)?.to_vec())
    }

    pub fn i64(&self, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
        let tensor = self.info(name)?;
        if tensor.dtype != "I64" || tensor.shape != shape {
            bail!(
                "{name}: expected I64{shape:?}, found {}{:?}",
                tensor.dtype,
                tensor.shape
            );
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.info(name)?;
        if tensor.dtype != "F32" || tensor.shape != shape {
            bail!(
                "{name}: expected F32{shape:?}, found {}{:?}",
                tensor.dtype,
                tensor.shape
            );
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }
}
