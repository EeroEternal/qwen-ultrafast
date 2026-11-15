use anyhow::{anyhow, bail, Context, Result};
pub use cudarc::driver::sys;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};

pub type DevicePtr = u64;

macro_rules! ck {
    ($expr:expr) => {
        check(unsafe { $expr }, stringify!($expr))
    };
}

pub fn check(result: sys::CUresult, operation: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut message = null();
    unsafe { sys::cuGetErrorString(result, &mut message) };
    let detail = if message.is_null() {
        "unknown CUDA error".into()
    } else {
        unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(anyhow!("{operation}: {result:?} ({detail})"))
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_count: i32,
}

pub struct Device {
    ordinal: i32,
    handle: sys::CUdevice,
    context: sys::CUcontext,
}

impl Device {
    pub fn new(ordinal: i32) -> Result<Self> {
        ck!(sys::cuInit(0))?;
        let mut handle = 0;
        ck!(sys::cuDeviceGet(&mut handle, ordinal))?;
        let mut context = null_mut();
        ck!(sys::cuDevicePrimaryCtxRetain(&mut context, handle))?;
        let device = Self {
            ordinal,
            handle,
            context,
        };
        device.bind()?;
        let info = device.info()?;
        if info.compute_capability != (12, 0) {
            bail!(
                "Qwen Ultrafast requires sm_120; GPU {ordinal} is sm_{}{}",
                info.compute_capability.0,
                info.compute_capability.1
            );
        }
        Ok(device)
    }

    pub fn bind(&self) -> Result<()> {
        ck!(sys::cuCtxSetCurrent(self.context))
    }

    pub fn context(&self) -> sys::CUcontext {
        self.context
    }

    pub fn enable_peer_access(&self, peer: usize) -> Result<()> {
        let result = unsafe { sys::cuCtxEnablePeerAccess(peer as sys::CUcontext, 0) };
        if result == sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Ok(());
        }
        check(result, "cuCtxEnablePeerAccess")
    }

    pub fn info(&self) -> Result<DeviceInfo> {
        let mut name = vec![0 as std::ffi::c_char; 256];
        ck!(sys::cuDeviceGetName(
            name.as_mut_ptr(),
            name.len() as i32,
            self.handle
        ))?;
        let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let (mut major, mut minor, mut sm_count) = (0, 0, 0);
        ck!(sys::cuDeviceGetAttribute(
            &mut major,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            self.handle
        ))?;
        ck!(sys::cuDeviceGetAttribute(
            &mut minor,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            self.handle
        ))?;
        ck!(sys::cuDeviceGetAttribute(
            &mut sm_count,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            self.handle
        ))?;
        Ok(DeviceInfo {
            ordinal: self.ordinal,
            name,
            compute_capability: (major, minor),
            sm_count,
        })
    }
}

pub fn can_access_peer(ordinal: i32, peer: i32) -> Result<bool> {
    ck!(sys::cuInit(0))?;
    let (mut device, mut peer_device) = (0, 0);
    ck!(sys::cuDeviceGet(&mut device, ordinal))?;
    ck!(sys::cuDeviceGet(&mut peer_device, peer))?;
    let mut accessible = 0;
    ck!(sys::cuDeviceCanAccessPeer(
        &mut accessible,
        device,
        peer_device
    ))?;
    Ok(accessible != 0)
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.handle) };
    }
}

pub struct Buffer<T> {
    ptr: DevicePtr,
    len: usize,
    marker: PhantomData<T>,
}

impl<T: Copy> Buffer<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .context("buffer size overflow")?;
        let mut ptr = 0;
        ck!(sys::cuMemAlloc_v2(&mut ptr, bytes.max(1)))?;
        Ok(Self {
            ptr,
            len,
            marker: PhantomData,
        })
    }

    pub fn from_host(data: &[T]) -> Result<Self> {
        let buffer = Self::alloc(data.len())?;
        buffer.upload(data)?;
        Ok(buffer)
    }

    pub fn upload(&self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            bail!("upload length {} != buffer length {}", data.len(), self.len);
        }
        if !data.is_empty() {
            ck!(sys::cuMemcpyHtoD_v2(
                self.ptr,
                data.as_ptr().cast(),
                std::mem::size_of_val(data)
            ))?;
        }
        Ok(())
    }

    pub fn download(&self) -> Result<Vec<T>>
    where
        T: Default,
    {
        let mut data = vec![T::default(); self.len];
        if !data.is_empty() {
            ck!(sys::cuMemcpyDtoH_v2(
                data.as_mut_ptr().cast(),
                self.ptr,
                std::mem::size_of_val(&data[..])
            ))?;
        }
        Ok(data)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { sys::cuMemFree_v2(self.ptr) };
    }
}

pub struct Stream {
    raw: sys::CUstream,
}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuStreamCreate(
            &mut raw,
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32
        ))?;
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<()> {
        ck!(sys::cuStreamSynchronize(self.raw))
    }

    pub fn raw(&self) -> sys::CUstream {
        self.raw
    }

    pub fn begin_capture(&self) -> Result<()> {
        ck!(sys::cuStreamBeginCapture_v2(
            self.raw,
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        ))
    }

    pub fn end_capture(&self) -> Result<Graph> {
        let mut raw = null_mut();
        ck!(sys::cuStreamEndCapture(self.raw, &mut raw))?;
        Ok(Graph { raw })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::cuStreamDestroy_v2(self.raw) };
    }
}

pub struct Event {
    raw: sys::CUevent,
}

impl Event {
    pub fn timed() -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuEventCreate(
            &mut raw,
            sys::CUevent_flags::CU_EVENT_DEFAULT as u32
        ))?;
        Ok(Self { raw })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        ck!(sys::cuEventRecord(self.raw, stream.raw))
    }

    pub fn synchronize(&self) -> Result<()> {
        ck!(sys::cuEventSynchronize(self.raw))
    }

    pub fn elapsed_ms(&self, start: &Event) -> Result<f32> {
        let mut elapsed = 0.0;
        ck!(sys::cuEventElapsedTime_v2(
            &mut elapsed,
            start.raw,
            self.raw
        ))?;
        Ok(elapsed)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::cuEventDestroy_v2(self.raw) };
    }
}

pub struct Graph {
    raw: sys::CUgraph,
}

impl Graph {
    pub fn instantiate(self) -> Result<GraphExec> {
        let mut raw = null_mut();
        ck!(sys::cuGraphInstantiateWithFlags(&mut raw, self.raw, 0))?;
        Ok(GraphExec { raw })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::cuGraphDestroy(self.raw) };
    }
}

pub struct GraphExec {
    raw: sys::CUgraphExec,
}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        ck!(sys::cuGraphLaunch(self.raw, stream.raw))
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { sys::cuGraphExecDestroy(self.raw) };
    }
}

pub struct Module {
    raw: sys::CUmodule,
}

impl Module {
    pub fn load(image: &[u8]) -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuModuleLoadData(&mut raw, image.as_ptr().cast()))?;
        Ok(Self { raw })
    }

    pub fn function(&self, name: &str) -> Result<sys::CUfunction> {
        let name = CString::new(name)?;
        let mut function = null_mut();
        ck!(sys::cuModuleGetFunction(
            &mut function,
            self.raw,
            name.as_ptr()
        ))?;
        Ok(function)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.raw) };
    }
}

#[derive(Clone, Copy)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_bytes: u32,
}

pub enum Argument {
    U32(u32),
    U64(u64),
    F32(f32),
}

#[derive(Default)]
pub struct Args(Vec<Argument>);

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(mut self, value: u32) -> Self {
        self.0.push(Argument::U32(value));
        self
    }

    pub fn u64(mut self, value: u64) -> Self {
        self.0.push(Argument::U64(value));
        self
    }

    pub fn f32(mut self, value: f32) -> Self {
        self.0.push(Argument::F32(value));
        self
    }

    fn pointers(&mut self) -> Vec<*mut c_void> {
        self.0
            .iter_mut()
            .map(|argument| match argument {
                Argument::U32(value) => (value as *mut u32).cast(),
                Argument::U64(value) => (value as *mut u64).cast(),
                Argument::F32(value) => (value as *mut f32).cast(),
            })
            .collect()
    }
}

/// # Safety
/// `args` must exactly match the function's CUDA parameter types and every
/// device pointer must remain valid until execution completes.
pub unsafe fn launch(
    function: sys::CUfunction,
    config: Launch,
    stream: &Stream,
    args: &mut Args,
) -> Result<()> {
    let mut pointers = args.pointers();
    check(
        sys::cuLaunchKernel(
            function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_bytes,
            stream.raw,
            pointers.as_mut_ptr(),
            null_mut(),
        ),
        "cuLaunchKernel",
    )
}
