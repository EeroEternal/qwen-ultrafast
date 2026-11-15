use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let mut entries = Vec::new();
    for stem in ["qwen", "gdn", "moe", "hc", "qsa", "ple"] {
        let cubin = out.join(format!("{stem}.cubin"));
        let status = Command::new(&nvcc)
            .args([
                "-arch=sm_120",
                "-cubin",
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "-Icu",
            ])
            .arg(format!("cu/{stem}.cu"))
            .arg("-o")
            .arg(&cubin)
            .status()
            .expect("failed to execute nvcc; set NVCC to the CUDA 13 compiler");
        assert!(status.success(), "nvcc failed for cu/{stem}.cu");
        entries.push(format!(
            "({stem:?}, include_bytes!({:?}) as &[u8])",
            cubin.to_str().unwrap()
        ));
    }
    fs::write(
        out.join("cubins.rs"),
        format!(
            "pub const CUBINS: &[(&str, &[u8])] = &[{}];\n",
            entries.join(",")
        ),
    )
    .unwrap();
}
