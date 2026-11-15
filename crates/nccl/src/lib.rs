use anyhow::{bail, Context, Result};
use libloading::Library;
use qwen_ultrafast_cuda::{Buffer, Stream};
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

const NCCL_SUCCESS: i32 = 0;
const NCCL_FLOAT32: i32 = 7;
const NCCL_BFLOAT16: i32 = 9;
const NCCL_SUM: i32 = 0;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct UniqueId {
    bytes: [u8; 128],
}

impl Default for UniqueId {
    fn default() -> Self {
        Self { bytes: [0; 128] }
    }
}

struct Api {
    _library: Library,
    get_unique_id: unsafe extern "C" fn(*mut UniqueId) -> i32,
    comm_init_rank: unsafe extern "C" fn(*mut *mut c_void, i32, UniqueId, i32) -> i32,
    all_reduce: unsafe extern "C" fn(u64, u64, usize, i32, i32, *mut c_void, *mut c_void) -> i32,
    comm_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    get_error_string: unsafe extern "C" fn(i32) -> *const c_char,
}

impl Api {
    fn load(path: &str) -> Result<Arc<Self>> {
        let library =
            unsafe { Library::new(path) }.with_context(|| format!("load NCCL library {path}"))?;
        unsafe {
            Ok(Arc::new(Self {
                get_unique_id: *library.get(b"ncclGetUniqueId")?,
                comm_init_rank: *library.get(b"ncclCommInitRank")?,
                all_reduce: *library.get(b"ncclAllReduce")?,
                comm_destroy: *library.get(b"ncclCommDestroy")?,
                get_error_string: *library.get(b"ncclGetErrorString")?,
                _library: library,
            }))
        }
    }

    fn load_default() -> Result<Arc<Self>> {
        if let Ok(path) = std::env::var("QWEN_NCCL_LIBRARY") {
            return Self::load(&path);
        }
        for path in ["libnccl.so.2", "/usr/lib/x86_64-linux-gnu/libnccl.so.2"] {
            if let Ok(api) = Self::load(path) {
                return Ok(api);
            }
        }
        bail!("NCCL not found; set QWEN_NCCL_LIBRARY to libnccl.so.2")
    }

    fn check(&self, result: i32) -> Result<()> {
        if result == NCCL_SUCCESS {
            return Ok(());
        }
        let message = unsafe { (self.get_error_string)(result) };
        let detail = if message.is_null() {
            "unknown NCCL error".into()
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("NCCL error {result}: {detail}")
    }
}

pub struct Comm {
    api: Arc<Api>,
    handle: *mut c_void,
}

impl Comm {
    pub fn unique_id() -> Result<UniqueId> {
        let api = Api::load_default()?;
        let mut id = UniqueId::default();
        api.check(unsafe { (api.get_unique_id)(&mut id) })?;
        Ok(id)
    }

    pub fn new(world_size: usize, rank: usize, id: UniqueId) -> Result<Self> {
        let api = Api::load_default()?;
        let mut handle = std::ptr::null_mut();
        api.check(unsafe {
            (api.comm_init_rank)(&mut handle, world_size as i32, id, rank as i32)
        })?;
        Ok(Self { api, handle })
    }

    pub fn all_reduce(&self, stream: &Stream, buffer: &Buffer<f32>) -> Result<()> {
        self.api.check(unsafe {
            (self.api.all_reduce)(
                buffer.ptr(),
                buffer.ptr(),
                buffer.len(),
                NCCL_FLOAT32,
                NCCL_SUM,
                self.handle,
                stream.raw().cast(),
            )
        })
    }

    pub fn all_reduce_bf16(&self, stream: &Stream, buffer: &Buffer<half::bf16>) -> Result<()> {
        self.api.check(unsafe {
            (self.api.all_reduce)(
                buffer.ptr(),
                buffer.ptr(),
                buffer.len(),
                NCCL_BFLOAT16,
                NCCL_SUM,
                self.handle,
                stream.raw().cast(),
            )
        })
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        unsafe { (self.api.comm_destroy)(self.handle) };
    }
}
