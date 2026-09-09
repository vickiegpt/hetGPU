use super::iq1s_tmatmul::CapturedActivationLaunch;
use super::iq1s_weight_registry::{
    global_registry, Iq1sExpertRole, ResolvedIq1sWeight, HETGPU_IQ1S_ERROR, HETGPU_IQ1S_HANDLED,
};
use std::collections::{BTreeSet, HashMap};
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const HETGPU_IQ1S_LAYER_ABI_VERSION: u32 = 2;
pub(crate) const HETGPU_IQ1S_PHASE_A: u32 = 1;
pub(crate) const QWEN35_EXPERT_COUNT: u32 = 512;
pub(crate) const QWEN35_EXPERTS_PER_TOKEN: u32 = 10;
pub(crate) const QWEN35_LAYER_COUNT: u32 = 60;
const MAX_LAYER_BATCH: u16 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerState {
    Open,
    RoutesPending,
    PhaseACapture,
    PhaseACommitted,
    PhaseADone,
    PhaseBCapture,
    PhaseBCommitted,
    Closed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LayerKey {
    pub(crate) session_generation: u64,
    pub(crate) transaction_id: u64,
    pub(crate) layer_id: u32,
    pub(crate) stream: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RouteAssignment {
    pub(crate) token_id: u32,
    pub(crate) expert_id: u16,
    pub(crate) route_weight: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct CapturedProjection {
    pub(crate) role: Iq1sExpertRole,
    pub(crate) weight: ResolvedIq1sWeight,
    pub(crate) launches: Vec<CapturedActivationLaunch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenLayerTransaction {
    key: LayerKey,
    batch_count: u16,
}

pub(crate) struct LayerTransaction {
    pub(crate) key: LayerKey,
    pub(crate) state: LayerState,
    pub(crate) batch_count: u16,
    pub(crate) routes: Vec<RouteAssignment>,
    pub(crate) projections: HashMap<Iq1sExpertRole, CapturedProjection>,
    pub(crate) expected_iq1s_roles: BTreeSet<Iq1sExpertRole>,
    pending_routes: Option<Box<dyn PendingRouteCopy>>,
    phase_b_committed: bool,
}

impl std::fmt::Debug for LayerTransaction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LayerTransaction")
            .field("key", &self.key)
            .field("state", &self.state)
            .field("batch_count", &self.batch_count)
            .field("routes", &self.routes)
            .field("projections", &self.projections)
            .field("expected_iq1s_roles", &self.expected_iq1s_roles)
            .field("routes_pending", &self.pending_routes.is_some())
            .field("phase_b_committed", &self.phase_b_committed)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RouteDmaRequest {
    pub(crate) token_ids: *const u32,
    pub(crate) expert_ids: *const i32,
    pub(crate) route_weights: *const f32,
    pub(crate) assignment_count: usize,
    pub(crate) stream: usize,
}

unsafe impl Send for RouteDmaRequest {}
unsafe impl Sync for RouteDmaRequest {}

pub(crate) trait PendingRouteCopy: Send {
    fn finish(self: Box<Self>) -> Result<Vec<RouteAssignment>, String>;
}

pub(crate) trait RouteDma: Send + Sync {
    unsafe fn enqueue(&self, request: RouteDmaRequest)
        -> Result<Box<dyn PendingRouteCopy>, String>;
}

type CuMemAllocHostFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CuMemFreeHostFn = unsafe extern "C" fn(*mut c_void) -> i32;

const CU_MEMCPY_SRC_ACCESS_ORDER_STREAM: i32 = 1;
const CU_MEM_LOCATION_TYPE_INVALID: i32 = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CuMemLocation {
    location_type: i32,
    id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CuMemcpyAttributes {
    src_access_order: i32,
    src_loc_hint: CuMemLocation,
    dst_loc_hint: CuMemLocation,
    flags: u32,
}

fn cuda_stream_ordered_copy_attributes() -> CuMemcpyAttributes {
    CuMemcpyAttributes {
        src_access_order: CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        src_loc_hint: CuMemLocation {
            location_type: CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        dst_loc_hint: CuMemLocation {
            location_type: CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        flags: 0,
    }
}

type CuMemcpyBatchAsyncFn = unsafe extern "C" fn(
    *mut *mut c_void,
    *mut *mut c_void,
    *mut usize,
    usize,
    *mut CuMemcpyAttributes,
    *mut usize,
    usize,
    *mut c_void,
) -> i32;
type CuEventCreateFn = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CuEventRecordFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type CuEventSynchronizeFn = unsafe extern "C" fn(*mut c_void) -> i32;
type CuEventDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct CudaRouteApi {
    _library: usize,
    mem_alloc_host: CuMemAllocHostFn,
    mem_free_host: CuMemFreeHostFn,
    memcpy_batch_async: CuMemcpyBatchAsyncFn,
    event_create: CuEventCreateFn,
    event_record: CuEventRecordFn,
    event_synchronize: CuEventSynchronizeFn,
    event_destroy: CuEventDestroyFn,
}

unsafe impl Send for CudaRouteApi {}
unsafe impl Sync for CudaRouteApi {}

fn cuda_symbol(handle: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let name = CString::new(name).map_err(|_| "CUDA symbol name contains NUL".to_string())?;
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr()) };
    if symbol.is_null() {
        return Err(format!("libcuda is missing {}", name.to_string_lossy()));
    }
    Ok(symbol)
}

fn load_cuda_route_api() -> Result<CudaRouteApi, String> {
    let mut handle = std::ptr::null_mut();
    for candidate in ["libcuda.so.1", "libcuda.so"] {
        let candidate = CString::new(candidate).expect("literal CUDA library name");
        handle = unsafe { libc::dlopen(candidate.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            break;
        }
    }
    if handle.is_null() {
        return Err("unable to load libcuda.so.1 for route-table DMA".to_string());
    }
    unsafe {
        Ok(CudaRouteApi {
            _library: handle as usize,
            mem_alloc_host: std::mem::transmute::<*mut c_void, CuMemAllocHostFn>(cuda_symbol(
                handle,
                "cuMemAllocHost_v2",
            )?),
            mem_free_host: std::mem::transmute::<*mut c_void, CuMemFreeHostFn>(cuda_symbol(
                handle,
                "cuMemFreeHost",
            )?),
            memcpy_batch_async: std::mem::transmute::<*mut c_void, CuMemcpyBatchAsyncFn>(
                cuda_symbol(handle, "cuMemcpyBatchAsync_v2")?,
            ),
            event_create: std::mem::transmute::<*mut c_void, CuEventCreateFn>(cuda_symbol(
                handle,
                "cuEventCreate",
            )?),
            event_record: std::mem::transmute::<*mut c_void, CuEventRecordFn>(cuda_symbol(
                handle,
                "cuEventRecord",
            )?),
            event_synchronize: std::mem::transmute::<*mut c_void, CuEventSynchronizeFn>(
                cuda_symbol(handle, "cuEventSynchronize")?,
            ),
            event_destroy: std::mem::transmute::<*mut c_void, CuEventDestroyFn>(cuda_symbol(
                handle,
                "cuEventDestroy_v2",
            )?),
        })
    }
}

fn cuda_route_api() -> Result<&'static CudaRouteApi, String> {
    static API: OnceLock<Result<CudaRouteApi, String>> = OnceLock::new();
    match API.get_or_init(load_cuda_route_api) {
        Ok(api) => Ok(api),
        Err(error) => Err(error.clone()),
    }
}

struct NativePendingRouteCopy {
    api: &'static CudaRouteApi,
    staging: usize,
    event: usize,
    assignment_count: usize,
    released: bool,
}

unsafe impl Send for NativePendingRouteCopy {}

impl NativePendingRouteCopy {
    unsafe fn synchronize_and_release(&mut self) -> Result<(), String> {
        let result = (self.api.event_synchronize)(self.event as *mut c_void);
        if result != 0 {
            // The DMA may still own the pinned buffer. Deliberately leak it rather than
            // permit use-after-free after a failed synchronization.
            self.released = true;
            return Err(format!("cuEventSynchronize failed with code {result}"));
        }
        let destroy = (self.api.event_destroy)(self.event as *mut c_void);
        let free = (self.api.mem_free_host)(self.staging as *mut c_void);
        self.released = true;
        if destroy != 0 {
            return Err(format!("cuEventDestroy_v2 failed with code {destroy}"));
        }
        if free != 0 {
            return Err(format!("cuMemFreeHost failed with code {free}"));
        }
        Ok(())
    }
}

impl PendingRouteCopy for NativePendingRouteCopy {
    fn finish(mut self: Box<Self>) -> Result<Vec<RouteAssignment>, String> {
        unsafe {
            let result = (self.api.event_synchronize)(self.event as *mut c_void);
            if result != 0 {
                self.released = true;
                return Err(format!("cuEventSynchronize failed with code {result}"));
            }
            let token_base = self.staging as *const u32;
            let expert_base = token_base.add(self.assignment_count) as *const i32;
            let weight_base = expert_base.add(self.assignment_count) as *const f32;
            let mut routes = Vec::with_capacity(self.assignment_count);
            for index in 0..self.assignment_count {
                let expert = expert_base.add(index).read_unaligned();
                let expert_id = u16::try_from(expert)
                    .map_err(|_| format!("route expert {expert} is outside u16"))?;
                routes.push(RouteAssignment {
                    token_id: token_base.add(index).read_unaligned(),
                    expert_id,
                    route_weight: weight_base.add(index).read_unaligned(),
                });
            }
            let destroy = (self.api.event_destroy)(self.event as *mut c_void);
            let free = (self.api.mem_free_host)(self.staging as *mut c_void);
            self.released = true;
            if destroy != 0 {
                return Err(format!("cuEventDestroy_v2 failed with code {destroy}"));
            }
            if free != 0 {
                return Err(format!("cuMemFreeHost failed with code {free}"));
            }
            Ok(routes)
        }
    }
}

impl Drop for NativePendingRouteCopy {
    fn drop(&mut self) {
        if !self.released {
            let _ = unsafe { self.synchronize_and_release() };
        }
    }
}

struct NativeCudaRouteDma;

impl RouteDma for NativeCudaRouteDma {
    unsafe fn enqueue(
        &self,
        request: RouteDmaRequest,
    ) -> Result<Box<dyn PendingRouteCopy>, String> {
        if request.token_ids.is_null()
            || request.expert_ids.is_null()
            || request.route_weights.is_null()
            || request.assignment_count == 0
            || request.stream == 0
        {
            return Err(
                "route-table CUDA pointers, stream, or extent are null or empty".to_string(),
            );
        }
        let api = cuda_route_api()?;
        let component_bytes = request
            .assignment_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("route-table component size overflow")?;
        let staging_bytes = component_bytes
            .checked_mul(3)
            .ok_or("route-table staging size overflow")?;
        let mut staging = std::ptr::null_mut();
        let alloc = (api.mem_alloc_host)(&mut staging, staging_bytes);
        if alloc != 0 || staging.is_null() {
            return Err(format!("cuMemAllocHost_v2 failed with code {alloc}"));
        }
        let mut event = std::ptr::null_mut();
        let create = (api.event_create)(&mut event, 2); // CU_EVENT_DISABLE_TIMING
        if create != 0 || event.is_null() {
            let _ = (api.mem_free_host)(staging);
            return Err(format!("cuEventCreate failed with code {create}"));
        }
        let stream = request.stream as *mut c_void;
        let mut destinations = [
            staging,
            staging.byte_add(component_bytes),
            staging.byte_add(component_bytes * 2),
        ];
        let mut sources = [
            request.token_ids.cast_mut().cast(),
            request.expert_ids.cast_mut().cast(),
            request.route_weights.cast_mut().cast(),
        ];
        let mut sizes = [component_bytes; 3];
        let mut attributes = [cuda_stream_ordered_copy_attributes()];
        let mut attribute_indices = [0usize];
        let batch = (api.memcpy_batch_async)(
            destinations.as_mut_ptr(),
            sources.as_mut_ptr(),
            sizes.as_mut_ptr(),
            destinations.len(),
            attributes.as_mut_ptr(),
            attribute_indices.as_mut_ptr(),
            attributes.len(),
            stream,
        );
        if batch != 0 {
            let record = (api.event_record)(event, stream);
            if record == 0 && (api.event_synchronize)(event) == 0 {
                let _ = (api.event_destroy)(event);
                let _ = (api.mem_free_host)(staging);
            }
            return Err(format!("cuMemcpyBatchAsync_v2 failed with code {batch}"));
        }
        let record = (api.event_record)(event, stream);
        if record != 0 {
            // Copies may be live and no reliable fence was recorded. Leak pinned staging.
            return Err(format!("cuEventRecord failed with code {record}"));
        }
        Ok(Box::new(NativePendingRouteCopy {
            api,
            staging: staging as usize,
            event: event as usize,
            assignment_count: request.assignment_count,
            released: false,
        }))
    }
}

pub(crate) struct PendingCudaOutputBatch {
    api: &'static CudaRouteApi,
    staging: usize,
    event: usize,
    released: bool,
}

unsafe impl Send for PendingCudaOutputBatch {}

impl PendingCudaOutputBatch {
    pub(crate) fn finish(mut self) -> Result<(), String> {
        unsafe { self.synchronize_and_release() }
    }

    unsafe fn synchronize_and_release(&mut self) -> Result<(), String> {
        let wait = (self.api.event_synchronize)(self.event as *mut c_void);
        if wait != 0 {
            self.released = true;
            return Err(format!(
                "cuEventSynchronize output batch failed with code {wait}"
            ));
        }
        let destroy = (self.api.event_destroy)(self.event as *mut c_void);
        let free = (self.api.mem_free_host)(self.staging as *mut c_void);
        self.released = true;
        if destroy != 0 {
            return Err(format!(
                "cuEventDestroy_v2 output batch failed with code {destroy}"
            ));
        }
        if free != 0 {
            return Err(format!(
                "cuMemFreeHost output batch failed with code {free}"
            ));
        }
        Ok(())
    }
}

impl Drop for PendingCudaOutputBatch {
    fn drop(&mut self) {
        if !self.released {
            let _ = unsafe { self.synchronize_and_release() };
        }
    }
}

pub(crate) unsafe fn enqueue_cuda_output_batch(
    stream: usize,
    outputs: &[(usize, &[u8])],
) -> Result<PendingCudaOutputBatch, String> {
    if stream == 0 || outputs.is_empty() {
        return Err("CUDA output batch requires a nonzero stream and outputs".to_string());
    }
    let total_bytes = outputs.iter().try_fold(0usize, |total, (dst, bytes)| {
        if *dst == 0 || bytes.is_empty() {
            return Err(
                "CUDA output batch contains a null destination or empty result".to_string(),
            );
        }
        total
            .checked_add(bytes.len())
            .ok_or_else(|| "CUDA output batch staging size overflow".to_string())
    })?;
    let api = cuda_route_api()?;
    let mut staging = std::ptr::null_mut();
    let alloc = (api.mem_alloc_host)(&mut staging, total_bytes);
    if alloc != 0 || staging.is_null() {
        return Err(format!(
            "cuMemAllocHost_v2 output batch failed with code {alloc}"
        ));
    }
    let mut event = std::ptr::null_mut();
    let create = (api.event_create)(&mut event, 2);
    if create != 0 || event.is_null() {
        let _ = (api.mem_free_host)(staging);
        return Err(format!(
            "cuEventCreate output batch failed with code {create}"
        ));
    }
    let mut destinations = Vec::with_capacity(outputs.len());
    let mut sources = Vec::with_capacity(outputs.len());
    let mut sizes = Vec::with_capacity(outputs.len());
    let mut cursor = 0usize;
    for (dst, bytes) in outputs {
        let source = staging.byte_add(cursor);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), source.cast::<u8>(), bytes.len());
        destinations.push(*dst as *mut c_void);
        sources.push(source);
        sizes.push(bytes.len());
        cursor += bytes.len();
    }
    let mut attributes = [cuda_stream_ordered_copy_attributes()];
    let mut attribute_indices = [0usize];
    let stream_ptr = stream as *mut c_void;
    let batch = (api.memcpy_batch_async)(
        destinations.as_mut_ptr(),
        sources.as_mut_ptr(),
        sizes.as_mut_ptr(),
        outputs.len(),
        attributes.as_mut_ptr(),
        attribute_indices.as_mut_ptr(),
        attributes.len(),
        stream_ptr,
    );
    if batch != 0 {
        let record = (api.event_record)(event, stream_ptr);
        if record == 0 && (api.event_synchronize)(event) == 0 {
            let _ = (api.event_destroy)(event);
            let _ = (api.mem_free_host)(staging);
        }
        return Err(format!(
            "cuMemcpyBatchAsync_v2 output batch failed with code {batch}"
        ));
    }
    let record = (api.event_record)(event, stream_ptr);
    if record != 0 {
        return Err(format!(
            "cuEventRecord output batch failed with code {record}"
        ));
    }
    Ok(PendingCudaOutputBatch {
        api,
        staging: staging as usize,
        event: event as usize,
        released: false,
    })
}

pub(crate) unsafe fn copy_cuda_to_host_batch(
    stream: usize,
    sources_and_sizes: &[(usize, usize)],
) -> Result<Vec<Vec<u8>>, String> {
    if stream == 0 || sources_and_sizes.is_empty() {
        return Err("CUDA input batch requires a nonzero stream and sources".to_string());
    }
    let total_bytes = sources_and_sizes
        .iter()
        .try_fold(0usize, |total, (src, bytes)| {
            if *src == 0 || *bytes == 0 {
                return Err("CUDA input batch contains a null source or empty extent".to_string());
            }
            total
                .checked_add(*bytes)
                .ok_or_else(|| "CUDA input batch staging size overflow".to_string())
        })?;
    let api = cuda_route_api()?;
    let mut staging = std::ptr::null_mut();
    let alloc = (api.mem_alloc_host)(&mut staging, total_bytes);
    if alloc != 0 || staging.is_null() {
        return Err(format!(
            "cuMemAllocHost_v2 input batch failed with code {alloc}"
        ));
    }
    let mut event = std::ptr::null_mut();
    let create = (api.event_create)(&mut event, 2);
    if create != 0 || event.is_null() {
        let _ = (api.mem_free_host)(staging);
        return Err(format!(
            "cuEventCreate input batch failed with code {create}"
        ));
    }
    let mut destinations = Vec::with_capacity(sources_and_sizes.len());
    let mut sources = Vec::with_capacity(sources_and_sizes.len());
    let mut sizes = Vec::with_capacity(sources_and_sizes.len());
    let mut cursor = 0usize;
    for (src, bytes) in sources_and_sizes {
        destinations.push(staging.byte_add(cursor));
        sources.push(*src as *mut c_void);
        sizes.push(*bytes);
        cursor += *bytes;
    }
    let mut attributes = [cuda_stream_ordered_copy_attributes()];
    let mut attribute_indices = [0usize];
    let stream_ptr = stream as *mut c_void;
    let batch = (api.memcpy_batch_async)(
        destinations.as_mut_ptr(),
        sources.as_mut_ptr(),
        sizes.as_mut_ptr(),
        sources_and_sizes.len(),
        attributes.as_mut_ptr(),
        attribute_indices.as_mut_ptr(),
        attributes.len(),
        stream_ptr,
    );
    if batch != 0 {
        let record = (api.event_record)(event, stream_ptr);
        if record == 0 && (api.event_synchronize)(event) == 0 {
            let _ = (api.event_destroy)(event);
            let _ = (api.mem_free_host)(staging);
        }
        return Err(format!(
            "cuMemcpyBatchAsync_v2 input batch failed with code {batch}"
        ));
    }
    let record = (api.event_record)(event, stream_ptr);
    if record != 0 {
        return Err(format!(
            "cuEventRecord input batch failed with code {record}"
        ));
    }
    let wait = (api.event_synchronize)(event);
    if wait != 0 {
        return Err(format!(
            "cuEventSynchronize input batch failed with code {wait}"
        ));
    }
    let mut copied = Vec::with_capacity(sources_and_sizes.len());
    cursor = 0;
    for (_, bytes) in sources_and_sizes {
        copied.push(
            std::slice::from_raw_parts(staging.byte_add(cursor).cast::<u8>(), *bytes).to_vec(),
        );
        cursor += *bytes;
    }
    let destroy = (api.event_destroy)(event);
    let free = (api.mem_free_host)(staging);
    if destroy != 0 {
        return Err(format!(
            "cuEventDestroy_v2 input batch failed with code {destroy}"
        ));
    }
    if free != 0 {
        return Err(format!("cuMemFreeHost input batch failed with code {free}"));
    }
    Ok(copied)
}

pub(crate) struct LayerCoordinator {
    session_generation: u64,
    top_k: u32,
    route_dma: Arc<dyn RouteDma>,
    transactions: Mutex<HashMap<(u64, u64), LayerTransaction>>,
}

impl LayerCoordinator {
    pub(crate) fn new(session_generation: u64, top_k: u32) -> Result<Self, String> {
        Self::with_route_dma(session_generation, top_k, Arc::new(NativeCudaRouteDma))
    }

    fn with_route_dma(
        session_generation: u64,
        top_k: u32,
        route_dma: Arc<dyn RouteDma>,
    ) -> Result<Self, String> {
        if session_generation == 0 {
            return Err("IQ1_S session generation must be nonzero".to_string());
        }
        if top_k == 0 || top_k > QWEN35_EXPERT_COUNT {
            return Err(format!("IQ1_S top_k must be in 1..={QWEN35_EXPERT_COUNT}"));
        }
        Ok(Self {
            session_generation,
            top_k,
            route_dma,
            transactions: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn begin(
        &self,
        layer_id: u32,
        transaction_id: u64,
        batch_count: u32,
        stream: usize,
        expected_iq1s_roles: BTreeSet<Iq1sExpertRole>,
    ) -> Result<(), String> {
        if transaction_id == 0 {
            return Err("IQ1_S transaction ID must be nonzero".to_string());
        }
        if stream == 0 {
            return Err("IQ1_S layer begin requires a non-legacy CUDA stream".to_string());
        }
        if layer_id >= QWEN35_LAYER_COUNT {
            return Err(format!(
                "IQ1_S layer {layer_id} is outside 0..{}",
                QWEN35_LAYER_COUNT - 1
            ));
        }
        let batch_count = u16::try_from(batch_count)
            .map_err(|_| "IQ1_S batch count is outside u16".to_string())?;
        if !(1..=MAX_LAYER_BATCH).contains(&batch_count) {
            return Err(format!(
                "IQ1_S batch count must be in 1..={MAX_LAYER_BATCH}, got {batch_count}"
            ));
        }
        let key = (self.session_generation, transaction_id);
        let mut transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        if let Some(existing) = transactions.get_mut(&key) {
            existing.state = LayerState::Aborted;
            return Err(format!("duplicate IQ1_S transaction ID {transaction_id}"));
        }
        if transactions.keys().any(|(generation, existing_id)| {
            *generation == self.session_generation && *existing_id > transaction_id
        }) {
            return Err(format!(
                "IQ1_S transaction ID {transaction_id} is not monotonically increasing"
            ));
        }
        transactions.insert(
            key,
            LayerTransaction {
                key: LayerKey {
                    session_generation: self.session_generation,
                    transaction_id,
                    layer_id,
                    stream,
                },
                state: LayerState::Open,
                batch_count,
                routes: Vec::new(),
                projections: HashMap::new(),
                expected_iq1s_roles,
                pending_routes: None,
                phase_b_committed: false,
            },
        );
        Ok(())
    }

    fn with_transaction<R>(
        &self,
        transaction_id: u64,
        operation: impl FnOnce(&mut LayerTransaction) -> Result<R, String>,
    ) -> Result<R, String> {
        let mut transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        let transaction = transactions
            .get_mut(&(self.session_generation, transaction_id))
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))?;
        if transaction.state == LayerState::Aborted {
            return Err(format!("IQ1_S transaction {transaction_id} is aborted"));
        }
        if matches!(transaction.state, LayerState::Closed) {
            return Err(format!(
                "IQ1_S transaction {transaction_id} is already closed"
            ));
        }
        match operation(transaction) {
            Ok(value) => Ok(value),
            Err(error) => {
                transaction.state = LayerState::Aborted;
                Err(error)
            }
        }
    }

    fn open_transaction_for_stream(
        &self,
        stream: usize,
    ) -> Result<Option<OpenLayerTransaction>, String> {
        if stream == 0 {
            return Ok(None);
        }
        let transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        let mut open = transactions
            .values()
            .filter(|transaction| {
                transaction.key.stream == stream
                    && !matches!(transaction.state, LayerState::Closed | LayerState::Aborted)
            })
            .map(|transaction| OpenLayerTransaction {
                key: transaction.key,
                batch_count: transaction.batch_count,
            });
        let first = open.next();
        if open.next().is_some() {
            return Err(format!(
                "CUDA stream {stream:#x} has more than one open IQ1_S layer transaction"
            ));
        }
        Ok(first)
    }

    fn validate_projection_routes(
        transaction: &LayerTransaction,
        projection: &CapturedProjection,
    ) -> Result<(), String> {
        if projection.launches.len() != transaction.routes.len() {
            return Err(format!(
                "IQ1_S projection has {} expert launches, expected {}",
                projection.launches.len(),
                transaction.routes.len()
            ));
        }
        let expert_stride = projection.weight.identity.nb[2];
        let first_offset = projection
            .weight
            .expert
            .checked_mul(expert_stride)
            .ok_or("IQ1_S first expert offset overflow")?;
        let tensor_base = (projection.launches[0].launch.matrix_ptr as u64)
            .checked_sub(first_offset)
            .ok_or("IQ1_S first expert pointer precedes its tensor base")?;
        for (launch, route) in projection.launches.iter().zip(&transaction.routes) {
            let relative = (launch.launch.matrix_ptr as u64)
                .checked_sub(tensor_base)
                .ok_or("IQ1_S expert launch pointer precedes its tensor base")?;
            if relative % expert_stride != 0 {
                return Err("IQ1_S expert launch pointer is not expert-aligned".to_string());
            }
            let expert = relative / expert_stride;
            if expert >= u64::from(QWEN35_EXPERT_COUNT) || expert != u64::from(route.expert_id) {
                return Err(format!(
                    "IQ1_S captured expert {expert} does not match routed expert {}",
                    route.expert_id
                ));
            }
        }
        Ok(())
    }

    fn validate_routes(
        &self,
        transaction: &LayerTransaction,
        routes: &[RouteAssignment],
    ) -> Result<(), String> {
        let expected_count = usize::from(transaction.batch_count)
            .checked_mul(self.top_k as usize)
            .ok_or("route assignment count overflow")?;
        if routes.len() != expected_count {
            return Err(format!(
                "IQ1_S route table has {} assignments, expected {expected_count}",
                routes.len()
            ));
        }
        let mut by_token = HashMap::<u32, BTreeSet<u16>>::new();
        for route in routes {
            if u32::from(route.expert_id) >= QWEN35_EXPERT_COUNT {
                return Err(format!(
                    "IQ1_S expert {} is outside 0..{}",
                    route.expert_id,
                    QWEN35_EXPERT_COUNT - 1
                ));
            }
            if !route.route_weight.is_finite() {
                return Err("IQ1_S route weight must be finite".to_string());
            }
            if !by_token
                .entry(route.token_id)
                .or_default()
                .insert(route.expert_id)
            {
                return Err(format!(
                    "IQ1_S token {} routes to expert {} more than once",
                    route.token_id, route.expert_id
                ));
            }
        }
        if by_token.len() != usize::from(transaction.batch_count)
            || by_token
                .values()
                .any(|experts| experts.len() != self.top_k as usize)
        {
            return Err(format!(
                "IQ1_S route table must contain exactly {} distinct experts for each of {} tokens",
                self.top_k, transaction.batch_count
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn set_routes_host(
        &self,
        transaction_id: u64,
        routes: Vec<RouteAssignment>,
    ) -> Result<(), String> {
        self.with_transaction(transaction_id, |transaction| {
            if transaction.state != LayerState::Open {
                return Err("IQ1_S routes may only be set once after begin".to_string());
            }
            self.validate_routes(transaction, &routes)?;
            transaction.routes = routes;
            transaction.state = LayerState::RoutesPending;
            Ok(())
        })
    }

    pub(crate) unsafe fn set_routes_device(
        &self,
        transaction_id: u64,
        token_ids: *const u32,
        expert_ids: *const i32,
        route_weights: *const f32,
        top_k: u32,
    ) -> Result<(), String> {
        if top_k != self.top_k {
            return self.with_transaction(transaction_id, |_| {
                Err(format!(
                    "IQ1_S top_k {top_k} does not match audited Qwen value {}",
                    self.top_k
                ))
            });
        }
        self.with_transaction(transaction_id, |transaction| {
            if transaction.state != LayerState::Open {
                return Err("IQ1_S routes may only be set once after begin".to_string());
            }
            let assignment_count = usize::from(transaction.batch_count)
                .checked_mul(self.top_k as usize)
                .ok_or("route assignment count overflow")?;
            let pending = self.route_dma.enqueue(RouteDmaRequest {
                token_ids,
                expert_ids,
                route_weights,
                assignment_count,
                stream: transaction.key.stream,
            })?;
            transaction.pending_routes = Some(pending);
            transaction.state = LayerState::RoutesPending;
            Ok(())
        })
    }

    fn finish_routes(&self, transaction_id: u64) -> Result<(), String> {
        let pending = self.with_transaction(transaction_id, |transaction| {
            if !matches!(
                transaction.state,
                LayerState::RoutesPending | LayerState::PhaseACapture
            ) {
                return Err("IQ1_S Phase A commit is out of order".to_string());
            }
            Ok(transaction.pending_routes.take())
        })?;
        let Some(pending) = pending else {
            return Ok(());
        };
        // The event wait deliberately runs without holding the coordinator map lock.
        let routes = pending.finish().map_err(|error| {
            let _ = self.with_transaction(transaction_id, |_| Err::<(), _>(error.clone()));
            error
        })?;
        self.with_transaction(transaction_id, |transaction| {
            self.validate_routes(transaction, &routes)?;
            transaction.routes = routes;
            Ok(())
        })
    }

    pub(crate) fn capture_projection(
        &self,
        session_generation: u64,
        transaction_id: u64,
        stream: usize,
        projection: CapturedProjection,
    ) -> Result<(), String> {
        self.with_transaction(transaction_id, |transaction| {
            if session_generation != transaction.key.session_generation {
                return Err("IQ1_S capture has a stale session generation".to_string());
            }
            if stream != transaction.key.stream {
                return Err("IQ1_S capture stream does not match layer begin".to_string());
            }
            if projection.role != projection.weight.identity.role {
                return Err("IQ1_S projection role does not match registered weight".to_string());
            }
            if projection.weight.identity.layer != transaction.key.layer_id {
                return Err("IQ1_S projection layer does not match layer begin".to_string());
            }
            if !transaction.expected_iq1s_roles.contains(&projection.role) {
                return Err("unexpected IQ1_S projection role for layer".to_string());
            }
            if projection.weight.expert >= u64::from(QWEN35_EXPERT_COUNT) {
                return Err("IQ1_S projection expert is outside 0..511".to_string());
            }
            if projection.launches.is_empty() {
                return Err("IQ1_S projection contains no captured launches".to_string());
            }
            for launch in &projection.launches {
                if launch.launch.allocation_generation != projection.weight.allocation_generation {
                    return Err("IQ1_S launch has a stale allocation generation".to_string());
                }
                if launch.launch.content_hash != projection.weight.content_sha256 {
                    return Err(
                        "IQ1_S launch content hash does not match registered weight".to_string()
                    );
                }
                let signature = &launch.launch.signature;
                if signature.ne00 != projection.weight.identity.ne[0]
                    || signature.ne01 != projection.weight.identity.ne[1]
                {
                    return Err("IQ1_S launch shape does not match registered weight".to_string());
                }
            }
            if transaction.projections.contains_key(&projection.role) {
                return Err("duplicate IQ1_S projection role in layer".to_string());
            }
            if !transaction.routes.is_empty() {
                Self::validate_projection_routes(transaction, &projection)?;
            }
            match projection.role {
                Iq1sExpertRole::Down => {
                    if transaction.state != LayerState::PhaseADone {
                        return Err(
                            "IQ1_S down projection arrived before Phase A completion".to_string()
                        );
                    }
                    transaction.state = LayerState::PhaseBCapture;
                }
                Iq1sExpertRole::Gate | Iq1sExpertRole::Up | Iq1sExpertRole::GateUp => {
                    if !matches!(
                        transaction.state,
                        LayerState::RoutesPending | LayerState::PhaseACapture
                    ) {
                        return Err(
                            "IQ1_S gate/up projection arrived outside Phase A capture".to_string()
                        );
                    }
                    transaction.state = LayerState::PhaseACapture;
                }
            }
            transaction.projections.insert(projection.role, projection);
            Ok(())
        })
    }

    pub(crate) fn commit_phase_a(&self, transaction_id: u64) -> Result<(), String> {
        self.finish_routes(transaction_id)?;
        self.with_transaction(transaction_id, |transaction| {
            if !matches!(transaction.state, LayerState::RoutesPending | LayerState::PhaseACapture) {
                return Err("IQ1_S Phase A commit is out of order".to_string());
            }
            if transaction.routes.is_empty() {
                return Err("IQ1_S Phase A commit has no completed route table".to_string());
            }
            let expected = transaction
                .expected_iq1s_roles
                .iter()
                .copied()
                .filter(|role| *role != Iq1sExpertRole::Down)
                .collect::<BTreeSet<_>>();
            let captured = transaction
                .projections
                .keys()
                .copied()
                .filter(|role| *role != Iq1sExpertRole::Down)
                .collect::<BTreeSet<_>>();
            if captured != expected {
                return Err(format!(
                    "IQ1_S Phase A roles are incomplete: expected {expected:?}, captured {captured:?}"
                ));
            }
            for projection in transaction.projections.values() {
                Self::validate_projection_routes(transaction, projection)?;
            }
            transaction.state = LayerState::PhaseACommitted;
            Ok(())
        })
    }

    pub(crate) fn complete_phase_a(&self, transaction_id: u64) -> Result<(), String> {
        self.with_transaction(transaction_id, |transaction| {
            if transaction.state != LayerState::PhaseACommitted {
                return Err("IQ1_S Phase A completion is out of order".to_string());
            }
            transaction.state = LayerState::PhaseADone;
            Ok(())
        })
    }

    pub(crate) fn snapshot_phase(
        &self,
        transaction_id: u64,
        phase: super::iq1s_layer_trace::LayerPhase,
    ) -> Result<super::iq1s_persistent_runtime::PhaseSnapshot, String> {
        let transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        let transaction = transactions
            .get(&(self.session_generation, transaction_id))
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))?;
        let expected_state = match phase {
            super::iq1s_layer_trace::LayerPhase::PhaseA => LayerState::PhaseACommitted,
            super::iq1s_layer_trace::LayerPhase::PhaseB => LayerState::PhaseBCapture,
        };
        if transaction.state != expected_state || transaction.pending_routes.is_some() {
            return Err(format!(
                "IQ1_S {phase:?} snapshot is not ready from state {:?}",
                transaction.state
            ));
        }
        Self::snapshot_transaction(transaction, phase)
    }

    fn snapshot_transaction(
        transaction: &LayerTransaction,
        phase: super::iq1s_layer_trace::LayerPhase,
    ) -> Result<super::iq1s_persistent_runtime::PhaseSnapshot, String> {
        if transaction.pending_routes.is_some() {
            return Err(format!(
                "IQ1_S {phase:?} snapshot still has pending route DMA"
            ));
        }
        let projections = transaction
            .projections
            .values()
            .filter(|projection| match phase {
                super::iq1s_layer_trace::LayerPhase::PhaseA => {
                    matches!(projection.role, Iq1sExpertRole::Gate | Iq1sExpertRole::Up)
                }
                super::iq1s_layer_trace::LayerPhase::PhaseB => {
                    projection.role == Iq1sExpertRole::Down
                }
            })
            .cloned()
            .collect::<Vec<_>>();
        if projections.is_empty() {
            return Err(format!("IQ1_S {phase:?} snapshot has no projections"));
        }
        Ok(super::iq1s_persistent_runtime::PhaseSnapshot {
            key: transaction.key,
            batch_count: transaction.batch_count,
            routes: transaction.routes.clone(),
            projections,
            phase,
        })
    }

    pub(crate) fn prepare_phase_a(
        &self,
        transaction_id: u64,
    ) -> Result<super::iq1s_persistent_runtime::PhaseSnapshot, String> {
        self.commit_phase_a(transaction_id)?;
        self.snapshot_phase(transaction_id, super::iq1s_layer_trace::LayerPhase::PhaseA)
    }

    pub(crate) fn prepare_phase_b(
        &self,
        transaction_id: u64,
    ) -> Result<super::iq1s_persistent_runtime::PhaseSnapshot, String> {
        self.with_transaction(transaction_id, |transaction| {
            if transaction.state != LayerState::PhaseBCapture
                || !transaction
                    .expected_iq1s_roles
                    .contains(&Iq1sExpertRole::Down)
                || !transaction.projections.contains_key(&Iq1sExpertRole::Down)
            {
                return Err("IQ1_S Phase B preparation is missing the down projection".to_string());
            }
            let snapshot = Self::snapshot_transaction(
                transaction,
                super::iq1s_layer_trace::LayerPhase::PhaseB,
            )?;
            transaction.state = LayerState::PhaseBCommitted;
            transaction.phase_b_committed = true;
            Ok(snapshot)
        })
    }

    pub(crate) fn complete_phase_b_and_close(&self, transaction_id: u64) -> Result<(), String> {
        self.with_transaction(transaction_id, |transaction| {
            if transaction.state != LayerState::PhaseBCommitted || !transaction.phase_b_committed {
                return Err("IQ1_S Phase B completion is out of order".to_string());
            }
            transaction.state = LayerState::Closed;
            Ok(())
        })
    }

    pub(crate) fn close_gpu_down(&self, transaction_id: u64) -> Result<(), String> {
        self.with_transaction(transaction_id, |transaction| {
            if transaction
                .expected_iq1s_roles
                .contains(&Iq1sExpertRole::Down)
                || transaction.state != LayerState::PhaseADone
            {
                return Err("GPU-native down may close only after Phase A completion".to_string());
            }
            transaction.state = LayerState::Closed;
            Ok(())
        })
    }

    fn expects_down(&self, transaction_id: u64) -> Result<bool, String> {
        let transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        let transaction = transactions
            .get(&(self.session_generation, transaction_id))
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))?;
        if matches!(transaction.state, LayerState::Closed | LayerState::Aborted) {
            return Err(format!(
                "IQ1_S transaction {transaction_id} is already terminal"
            ));
        }
        Ok(transaction
            .expected_iq1s_roles
            .contains(&Iq1sExpertRole::Down))
    }

    pub(crate) fn commit_layer(&self, transaction_id: u64) -> Result<(), String> {
        if self.expects_down(transaction_id)? {
            self.prepare_phase_b(transaction_id)?;
            self.complete_phase_b_and_close(transaction_id)
        } else {
            self.close_gpu_down(transaction_id)
        }
    }

    pub(crate) fn abort(&self, transaction_id: u64, _reason: u32) -> Result<(), String> {
        let mut transactions = self
            .transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?;
        let transaction = transactions
            .get_mut(&(self.session_generation, transaction_id))
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))?;
        if matches!(transaction.state, LayerState::Closed | LayerState::Aborted) {
            return Err(format!(
                "IQ1_S transaction {transaction_id} is already terminal"
            ));
        }
        transaction.state = LayerState::Aborted;
        Ok(())
    }

    #[cfg(test)]
    fn state(&self, transaction_id: u64) -> Result<LayerState, String> {
        self.transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?
            .get(&(self.session_generation, transaction_id))
            .map(|transaction| transaction.state)
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))
    }

    #[cfg(test)]
    fn phase_b_committed(&self, transaction_id: u64) -> Result<bool, String> {
        self.transactions
            .lock()
            .map_err(|_| "IQ1_S layer coordinator lock poisoned".to_string())?
            .get(&(self.session_generation, transaction_id))
            .map(|transaction| transaction.phase_b_committed)
            .ok_or_else(|| format!("unknown IQ1_S transaction ID {transaction_id}"))
    }
}

fn global_coordinator() -> &'static LayerCoordinator {
    static COORDINATOR: OnceLock<LayerCoordinator> = OnceLock::new();
    COORDINATOR.get_or_init(|| {
        LayerCoordinator::new(1, QWEN35_EXPERTS_PER_TOKEN)
            .expect("constant IQ1_S coordinator configuration")
    })
}

trait Iq1sLayerPhaseExecutor {
    fn execute_phase(
        &mut self,
        snapshot: super::iq1s_persistent_runtime::PhaseSnapshot,
    ) -> Result<(), String>;

    fn poison(&mut self, error: &str);
}

struct GlobalPersistentPhaseExecutor;

impl Iq1sLayerPhaseExecutor for GlobalPersistentPhaseExecutor {
    fn execute_phase(
        &mut self,
        snapshot: super::iq1s_persistent_runtime::PhaseSnapshot,
    ) -> Result<(), String> {
        super::iq1s_persistent_runtime::execute_global_persistent_phase(snapshot)
    }

    fn poison(&mut self, error: &str) {
        super::iq1s_persistent_runtime::poison_global_persistent_runtime(error);
    }
}

fn execute_transaction_phase_with(
    coordinator: &LayerCoordinator,
    executor: &mut impl Iq1sLayerPhaseExecutor,
    transaction_id: u64,
    phase: super::iq1s_layer_trace::LayerPhase,
) -> Result<(), String> {
    let result = (|| {
        let snapshot = match phase {
            super::iq1s_layer_trace::LayerPhase::PhaseA => {
                coordinator.prepare_phase_a(transaction_id)?
            }
            super::iq1s_layer_trace::LayerPhase::PhaseB => {
                coordinator.prepare_phase_b(transaction_id)?
            }
        };
        executor.execute_phase(snapshot)?;
        match phase {
            super::iq1s_layer_trace::LayerPhase::PhaseA => {
                coordinator.complete_phase_a(transaction_id)
            }
            super::iq1s_layer_trace::LayerPhase::PhaseB => {
                coordinator.complete_phase_b_and_close(transaction_id)
            }
        }
    })();
    if let Err(error) = result {
        let _ = coordinator.abort(transaction_id, 0);
        executor.poison(&error);
        return Err(error);
    }
    Ok(())
}

fn commit_transaction_with(
    coordinator: &LayerCoordinator,
    executor: &mut impl Iq1sLayerPhaseExecutor,
    transaction_id: u64,
) -> Result<(), String> {
    let result = (|| {
        if coordinator.expects_down(transaction_id)? {
            execute_transaction_phase_with(
                coordinator,
                executor,
                transaction_id,
                super::iq1s_layer_trace::LayerPhase::PhaseB,
            )
        } else {
            coordinator.close_gpu_down(transaction_id)
        }
    })();
    if let Err(error) = result {
        let _ = coordinator.abort(transaction_id, 0);
        executor.poison(&error);
        return Err(error);
    }
    Ok(())
}

fn execute_transaction_phase(
    transaction_id: u64,
    phase: super::iq1s_layer_trace::LayerPhase,
) -> Result<(), String> {
    execute_transaction_phase_with(
        global_coordinator(),
        &mut GlobalPersistentPhaseExecutor,
        transaction_id,
        phase,
    )
}

pub(crate) fn has_open_transaction(stream: usize) -> Result<bool, String> {
    Ok(global_coordinator()
        .open_transaction_for_stream(stream)?
        .is_some())
}

pub(crate) fn open_layer_key(stream: usize) -> Result<Option<LayerKey>, String> {
    Ok(global_coordinator()
        .open_transaction_for_stream(stream)?
        .map(|open| open.key))
}

pub(crate) fn resolve_captured_role(
    launches: &[CapturedActivationLaunch],
) -> Result<Iq1sExpertRole, String> {
    let launch = launches
        .first()
        .ok_or("IQ1_S projection contains no captured launches")?;
    let signature = &launch.launch.signature;
    let matrix_bytes = u64::try_from(signature.matrix_storage_bytes()?)
        .map_err(|_| "IQ1_S matrix bytes do not fit u64")?;
    let row_stride = signature
        .stride01
        .checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
        .ok_or("IQ1_S row stride overflow")?;
    let resolved = global_registry().resolve_launch(
        launch.launch.matrix_ptr,
        matrix_bytes,
        signature.ne00,
        signature.ne01,
        row_stride,
    )?;
    Ok(resolved.identity.role)
}

fn resolve_registered_projection(
    role: Iq1sExpertRole,
    launches: Vec<CapturedActivationLaunch>,
) -> Result<CapturedProjection, String> {
    if launches.is_empty() {
        return Err("IQ1_S projection contains no captured launches".to_string());
    }
    let mut first_weight: Option<ResolvedIq1sWeight> = None;
    let mut experts = Vec::with_capacity(launches.len());
    for launch in &launches {
        let signature = &launch.launch.signature;
        let matrix_bytes = u64::try_from(signature.matrix_storage_bytes()?)
            .map_err(|_| "IQ1_S matrix bytes do not fit u64")?;
        let row_stride = signature
            .stride01
            .checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
            .ok_or("IQ1_S row stride overflow")?;
        let resolved = global_registry().resolve_launch(
            launch.launch.matrix_ptr,
            matrix_bytes,
            signature.ne00,
            signature.ne01,
            row_stride,
        )?;
        if resolved.identity.role != role {
            return Err(format!(
                "IQ1_S registered role {:?} does not match captured role {role:?}",
                resolved.identity.role
            ));
        }
        if launch.launch.allocation_generation != resolved.allocation_generation
            || launch.launch.content_hash != resolved.content_sha256
        {
            return Err("IQ1_S captured launch identity is stale".to_string());
        }
        if let Some(first) = &first_weight {
            if first.identity != resolved.identity
                || first.allocation_generation != resolved.allocation_generation
                || first.content_sha256 != resolved.content_sha256
            {
                return Err("IQ1_S projection spans more than one registered tensor".to_string());
            }
        } else {
            first_weight = Some(resolved.clone());
        }
        experts.push(resolved.expert);
    }
    for token_experts in experts.chunks_exact(QWEN35_EXPERTS_PER_TOKEN as usize) {
        let distinct = token_experts.iter().copied().collect::<BTreeSet<_>>();
        if distinct.len() != QWEN35_EXPERTS_PER_TOKEN as usize {
            return Err("IQ1_S captured token contains a duplicate expert ID".to_string());
        }
    }
    if !experts
        .len()
        .is_multiple_of(QWEN35_EXPERTS_PER_TOKEN as usize)
    {
        return Err("IQ1_S projection launch count is not divisible by top_k=10".to_string());
    }
    Ok(CapturedProjection {
        role,
        weight: first_weight.expect("nonempty projection has a resolved weight"),
        launches,
    })
}

pub(crate) fn capture_projection(
    stream: usize,
    role: Iq1sExpertRole,
    launches: Vec<CapturedActivationLaunch>,
) -> Result<(), String> {
    let open = global_coordinator()
        .open_transaction_for_stream(stream)?
        .ok_or_else(|| format!("CUDA stream {stream:#x} has no open IQ1_S transaction"))?;
    let expected_launches = usize::from(open.batch_count)
        .checked_mul(QWEN35_EXPERTS_PER_TOKEN as usize)
        .ok_or("IQ1_S projection launch count overflow")?;
    if launches.len() != expected_launches {
        return Err(format!(
            "IQ1_S projection has {} expert launches, expected {expected_launches}",
            launches.len()
        ));
    }
    let projection = resolve_registered_projection(role, launches)?;
    global_coordinator().capture_projection(
        open.key.session_generation,
        open.key.transaction_id,
        stream,
        projection,
    )
}

pub(crate) unsafe fn layer_begin(
    abi_version: u32,
    layer_id: u32,
    transaction_id: u64,
    batch_count: u32,
    cuda_stream: *mut c_void,
) -> i32 {
    let result = (|| -> Result<(), String> {
        if abi_version != HETGPU_IQ1S_LAYER_ABI_VERSION {
            return Err(format!("unsupported IQ1_S layer ABI version {abi_version}"));
        }
        let expected = global_registry().expected_roles_for_layer(layer_id)?;
        global_coordinator().begin(
            layer_id,
            transaction_id,
            batch_count,
            cuda_stream as usize,
            expected,
        )
    })();
    ffi_result("begin", result)
}

pub(crate) unsafe fn layer_set_routes(
    abi_version: u32,
    transaction_id: u64,
    token_ids: *const u32,
    expert_ids: *const i32,
    route_weights: *const f32,
    top_k: u32,
) -> i32 {
    let result = if abi_version != HETGPU_IQ1S_LAYER_ABI_VERSION {
        Err(format!("unsupported IQ1_S layer ABI version {abi_version}"))
    } else {
        global_coordinator().set_routes_device(
            transaction_id,
            token_ids,
            expert_ids,
            route_weights,
            top_k,
        )
    };
    ffi_result("set_routes", result)
}

pub(crate) fn layer_phase_commit(abi_version: u32, transaction_id: u64, phase: u32) -> i32 {
    let result = (|| -> Result<(), String> {
        if abi_version != HETGPU_IQ1S_LAYER_ABI_VERSION {
            return Err(format!("unsupported IQ1_S layer ABI version {abi_version}"));
        }
        if phase != HETGPU_IQ1S_PHASE_A {
            return Err(format!("unsupported IQ1_S layer phase {phase}"));
        }
        execute_transaction_phase(transaction_id, super::iq1s_layer_trace::LayerPhase::PhaseA)
    })();
    ffi_result("phase_commit", result)
}

pub(crate) fn layer_commit(abi_version: u32, transaction_id: u64) -> i32 {
    let result = if abi_version != HETGPU_IQ1S_LAYER_ABI_VERSION {
        Err(format!("unsupported IQ1_S layer ABI version {abi_version}"))
    } else {
        commit_transaction_with(
            global_coordinator(),
            &mut GlobalPersistentPhaseExecutor,
            transaction_id,
        )
    };
    ffi_result("commit", result)
}

pub(crate) fn layer_abort(abi_version: u32, transaction_id: u64, reason: u32) -> i32 {
    let result = if abi_version != HETGPU_IQ1S_LAYER_ABI_VERSION {
        Err(format!("unsupported IQ1_S layer ABI version {abi_version}"))
    } else {
        global_coordinator().abort(transaction_id, reason)
    };
    ffi_result("abort", result)
}

fn ffi_result(operation: &str, result: Result<(), String>) -> i32 {
    match result {
        Ok(()) => HETGPU_IQ1S_HANDLED,
        Err(error) => {
            eprintln!("[hetgpu-iq1s-layer] {operation} failed: {error}");
            HETGPU_IQ1S_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::iq1s_tmatmul::{
        capture_activation_from_host, GgmlType19Signature, LogicalLaunch, Q8_1_MMQ_BYTES,
    };
    use crate::r#impl::iq1s_weight_registry::{
        Iq1sExpertRole, Iq1sTensorIdentity, ResolvedIq1sWeight,
    };
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MockPendingRoutes {
        waits: Arc<AtomicUsize>,
        routes: Vec<RouteAssignment>,
    }

    impl PendingRouteCopy for MockPendingRoutes {
        fn finish(self: Box<Self>) -> Result<Vec<RouteAssignment>, String> {
            self.waits.fetch_add(1, Ordering::SeqCst);
            Ok(self.routes)
        }
    }

    struct MockRouteDma {
        enqueues: Arc<AtomicUsize>,
        waits: Arc<AtomicUsize>,
        routes: Vec<RouteAssignment>,
    }

    impl RouteDma for MockRouteDma {
        unsafe fn enqueue(
            &self,
            _request: RouteDmaRequest,
        ) -> Result<Box<dyn PendingRouteCopy>, String> {
            self.enqueues.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(MockPendingRoutes {
                waits: self.waits.clone(),
                routes: self.routes.clone(),
            }))
        }
    }

    fn roles(values: &[Iq1sExpertRole]) -> BTreeSet<Iq1sExpertRole> {
        values.iter().copied().collect()
    }

    fn routes(batch_count: u16) -> Vec<RouteAssignment> {
        (0..u32::from(batch_count))
            .flat_map(|token_id| {
                (0..QWEN35_EXPERTS_PER_TOKEN).map(move |slot| RouteAssignment {
                    token_id,
                    expert_id: (token_id * QWEN35_EXPERTS_PER_TOKEN + slot) as u16,
                    route_weight: 0.1,
                })
            })
            .collect()
    }

    fn projection(
        layer: u32,
        role: Iq1sExpertRole,
        batch_count: u16,
        generation: u64,
    ) -> CapturedProjection {
        let role_tag = match role {
            Iq1sExpertRole::Gate => 1usize,
            Iq1sExpertRole::Up => 2,
            Iq1sExpertRole::Down => 3,
            Iq1sExpertRole::GateUp => 4,
        };
        let content_hash = [role_tag as u8; 32];
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".to_string(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let packed_activations = vec![0u8; 2 * Q8_1_MMQ_BYTES];
        let tensor_base = 0x10_0000 + role_tag * 0x1_0000;
        let launches = routes(batch_count)
            .into_iter()
            .enumerate()
            .map(|(route_index, route)| {
                let launch = LogicalLaunch {
                    matrix_ptr: tensor_base + usize::from(route.expert_id) * 50,
                    activation_ptr: 0x20_0000 + role_tag * 0x1_0000 + route_index * 0x100,
                    output_ptr: 0x30_0000 + role_tag * 0x1_0000 + route_index * 0x100,
                    allocation_generation: generation,
                    content_hash,
                    signature: signature.clone(),
                };
                capture_activation_from_host(launch, &packed_activations)
                    .expect("synthetic projection must be valid")
            })
            .collect();
        let name = match role {
            Iq1sExpertRole::Gate => "gate",
            Iq1sExpertRole::Up => "up",
            Iq1sExpertRole::Down => "down",
            Iq1sExpertRole::GateUp => "gate_up",
        };
        CapturedProjection {
            role,
            weight: ResolvedIq1sWeight {
                identity: Iq1sTensorIdentity {
                    canonical_path: PathBuf::from("/tmp/qwen35-test.gguf"),
                    file_offset: 0,
                    nbytes: 50 * QWEN35_EXPERT_COUNT as u64,
                    name: format!("blk.{layer}.ffn_{name}_exps.weight"),
                    layer,
                    ne: [256, 1, QWEN35_EXPERT_COUNT as u64, 1],
                    nb: [50, 50, 50, 50 * QWEN35_EXPERT_COUNT as u64],
                    role,
                    model_sha256: [0x11; 32],
                    content_sha256: content_hash,
                    device: 1,
                    inode: 2,
                    modified_ns: 3,
                },
                expert: 0,
                allocation_generation: generation,
                content_sha256: content_hash,
            },
            launches,
        }
    }

    fn begin_three_role_layer(coordinator: &LayerCoordinator, transaction_id: u64) {
        coordinator
            .begin(
                12,
                transaction_id,
                2,
                0xabc0,
                roles(&[
                    Iq1sExpertRole::Gate,
                    Iq1sExpertRole::Up,
                    Iq1sExpertRole::Down,
                ]),
            )
            .unwrap();
        coordinator
            .set_routes_host(transaction_id, routes(2))
            .unwrap();
    }

    #[test]
    fn full_layer_transaction_reaches_closed_in_order() {
        let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        begin_three_role_layer(&coordinator, 100);
        coordinator
            .capture_projection(7, 100, 0xabc0, projection(12, Iq1sExpertRole::Gate, 2, 7))
            .unwrap();
        coordinator
            .capture_projection(7, 100, 0xabc0, projection(12, Iq1sExpertRole::Up, 2, 7))
            .unwrap();
        coordinator.commit_phase_a(100).unwrap();
        assert_eq!(coordinator.state(100).unwrap(), LayerState::PhaseACommitted);
        coordinator.complete_phase_a(100).unwrap();
        coordinator
            .capture_projection(7, 100, 0xabc0, projection(12, Iq1sExpertRole::Down, 2, 7))
            .unwrap();
        coordinator.commit_layer(100).unwrap();
        assert_eq!(coordinator.state(100).unwrap(), LayerState::Closed);
    }

    #[derive(Default)]
    struct FakePhaseExecutor {
        phases: Vec<(u64, crate::r#impl::iq1s_layer_trace::LayerPhase)>,
        fail_once: Option<String>,
        poisoned: Option<String>,
    }

    impl Iq1sLayerPhaseExecutor for FakePhaseExecutor {
        fn execute_phase(
            &mut self,
            snapshot: crate::r#impl::iq1s_persistent_runtime::PhaseSnapshot,
        ) -> Result<(), String> {
            if let Some(error) = &self.poisoned {
                return Err(format!("fake persistent runtime is poisoned: {error}"));
            }
            self.phases
                .push((snapshot.key.transaction_id, snapshot.phase));
            if let Some(error) = self.fail_once.take() {
                return Err(error);
            }
            Ok(())
        }

        fn poison(&mut self, error: &str) {
            if self.poisoned.is_none() {
                self.poisoned = Some(error.to_string());
            }
        }
    }

    fn capture_phase_a(coordinator: &LayerCoordinator, transaction_id: u64) {
        begin_three_role_layer(coordinator, transaction_id);
        coordinator
            .capture_projection(
                coordinator.session_generation,
                transaction_id,
                0xabc0,
                projection(12, Iq1sExpertRole::Gate, 2, coordinator.session_generation),
            )
            .unwrap();
        coordinator
            .capture_projection(
                coordinator.session_generation,
                transaction_id,
                0xabc0,
                projection(12, Iq1sExpertRole::Up, 2, coordinator.session_generation),
            )
            .unwrap();
    }

    #[test]
    fn qwen_iq1s_layer_integration_submits_phase_a_and_b_once() {
        let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        let mut executor = FakePhaseExecutor::default();
        capture_phase_a(&coordinator, 110);

        execute_transaction_phase_with(
            &coordinator,
            &mut executor,
            110,
            crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
        )
        .unwrap();
        assert_eq!(coordinator.state(110).unwrap(), LayerState::PhaseADone);
        coordinator
            .capture_projection(7, 110, 0xabc0, projection(12, Iq1sExpertRole::Down, 2, 7))
            .unwrap();
        commit_transaction_with(&coordinator, &mut executor, 110).unwrap();

        assert_eq!(
            executor.phases,
            vec![
                (110, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA),
                (110, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseB),
            ]
        );
        assert_eq!(coordinator.state(110).unwrap(), LayerState::Closed);
    }

    #[test]
    fn qwen_iq1s_layer_integration_first_executor_fault_aborts_and_poisons_session() {
        for fault in [
            "completion identity mismatch",
            "persistent timeout",
            "result contains nonfinite f32",
        ] {
            let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
            let mut executor = FakePhaseExecutor {
                fail_once: Some(fault.to_string()),
                ..FakePhaseExecutor::default()
            };
            capture_phase_a(&coordinator, 120);
            let error = execute_transaction_phase_with(
                &coordinator,
                &mut executor,
                120,
                crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
            )
            .unwrap_err();
            assert!(error.contains(fault));
            assert_eq!(coordinator.state(120).unwrap(), LayerState::Aborted);

            capture_phase_a(&coordinator, 121);
            let later = execute_transaction_phase_with(
                &coordinator,
                &mut executor,
                121,
                crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
            )
            .unwrap_err();
            assert!(later.contains("poisoned"));
            assert_eq!(coordinator.state(121).unwrap(), LayerState::Aborted);
            assert_eq!(executor.phases.len(), 1);
        }
    }

    #[test]
    fn qwen_iq1s_layer_integration_missing_lifecycle_poisons_later_work() {
        let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        let mut executor = FakePhaseExecutor::default();
        let error = execute_transaction_phase_with(
            &coordinator,
            &mut executor,
            130,
            crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
        )
        .unwrap_err();
        assert!(error.contains("unknown IQ1_S transaction"), "{error}");
        assert!(executor.poisoned.is_some());

        capture_phase_a(&coordinator, 131);
        let later = execute_transaction_phase_with(
            &coordinator,
            &mut executor,
            131,
            crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
        )
        .unwrap_err();
        assert!(later.contains("poisoned"), "{later}");
        assert_eq!(coordinator.state(131).unwrap(), LayerState::Aborted);
        assert!(executor.phases.is_empty());
    }

    #[test]
    fn qwen_iq1s_layer_integration_phase_b_fault_does_not_close_transaction() {
        let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        let mut executor = FakePhaseExecutor::default();
        capture_phase_a(&coordinator, 140);
        execute_transaction_phase_with(
            &coordinator,
            &mut executor,
            140,
            crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA,
        )
        .unwrap();
        coordinator
            .capture_projection(7, 140, 0xabc0, projection(12, Iq1sExpertRole::Down, 2, 7))
            .unwrap();
        executor.fail_once = Some("Phase B completion identity mismatch".to_string());

        let error = commit_transaction_with(&coordinator, &mut executor, 140).unwrap_err();
        assert!(error.contains("identity mismatch"), "{error}");
        assert_eq!(coordinator.state(140).unwrap(), LayerState::Aborted);
        assert_eq!(
            executor.phases,
            vec![
                (140, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA),
                (140, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseB),
            ]
        );
    }

    #[test]
    fn iq1s_persistent_runtime_snapshots_phase_data_without_holding_state_lock() {
        let coordinator = LayerCoordinator::new(7, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        begin_three_role_layer(&coordinator, 102);
        coordinator
            .capture_projection(7, 102, 0xabc0, projection(12, Iq1sExpertRole::Gate, 2, 7))
            .unwrap();
        coordinator
            .capture_projection(7, 102, 0xabc0, projection(12, Iq1sExpertRole::Up, 2, 7))
            .unwrap();
        coordinator.commit_phase_a(102).unwrap();

        let phase_a = coordinator
            .snapshot_phase(102, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseA)
            .unwrap();
        assert_eq!(phase_a.key.transaction_id, 102);
        assert_eq!(phase_a.batch_count, 2);
        assert_eq!(phase_a.routes.len(), 2 * QWEN35_EXPERTS_PER_TOKEN as usize);
        assert_eq!(phase_a.projections.len(), 2);

        coordinator.complete_phase_a(102).unwrap();
        coordinator
            .capture_projection(7, 102, 0xabc0, projection(12, Iq1sExpertRole::Down, 2, 7))
            .unwrap();
        let phase_b = coordinator
            .snapshot_phase(102, crate::r#impl::iq1s_layer_trace::LayerPhase::PhaseB)
            .unwrap();
        assert_eq!(phase_b.projections.len(), 1);
        assert_eq!(phase_b.projections[0].role, Iq1sExpertRole::Down);

        coordinator.commit_layer(102).unwrap();
        assert_eq!(coordinator.state(102).unwrap(), LayerState::Closed);
    }

    #[test]
    fn device_routes_enqueue_at_set_and_wait_only_at_phase_a_commit() {
        let enqueues = Arc::new(AtomicUsize::new(0));
        let waits = Arc::new(AtomicUsize::new(0));
        let coordinator = LayerCoordinator::with_route_dma(
            8,
            QWEN35_EXPERTS_PER_TOKEN,
            Arc::new(MockRouteDma {
                enqueues: enqueues.clone(),
                waits: waits.clone(),
                routes: routes(1),
            }),
        )
        .unwrap();
        coordinator
            .begin(
                12,
                99,
                1,
                0xabc0,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        unsafe {
            coordinator
                .set_routes_device(
                    99,
                    0x1000usize as *const u32,
                    0x2000usize as *const i32,
                    0x3000usize as *const f32,
                    QWEN35_EXPERTS_PER_TOKEN,
                )
                .unwrap();
        }
        assert_eq!(enqueues.load(Ordering::SeqCst), 1);
        assert_eq!(waits.load(Ordering::SeqCst), 0);
        coordinator
            .capture_projection(8, 99, 0xabc0, projection(12, Iq1sExpertRole::Gate, 1, 8))
            .unwrap();
        coordinator
            .capture_projection(8, 99, 0xabc0, projection(12, Iq1sExpertRole::Up, 1, 8))
            .unwrap();
        assert_eq!(waits.load(Ordering::SeqCst), 0);
        coordinator.commit_phase_a(99).unwrap();
        assert_eq!(waits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn gpu_native_down_closes_without_phase_b_commit() {
        let coordinator = LayerCoordinator::new(9, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        coordinator
            .begin(
                3,
                101,
                1,
                0x1110,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        coordinator.set_routes_host(101, routes(1)).unwrap();
        coordinator
            .capture_projection(9, 101, 0x1110, projection(3, Iq1sExpertRole::Gate, 1, 9))
            .unwrap();
        coordinator
            .capture_projection(9, 101, 0x1110, projection(3, Iq1sExpertRole::Up, 1, 9))
            .unwrap();
        coordinator.commit_phase_a(101).unwrap();
        coordinator.complete_phase_a(101).unwrap();
        coordinator.commit_layer(101).unwrap();

        assert_eq!(coordinator.state(101).unwrap(), LayerState::Closed);
        assert!(!coordinator.phase_b_committed(101).unwrap());
    }

    #[test]
    fn rejects_invalid_batches_and_duplicate_transaction_ids() {
        let coordinator = LayerCoordinator::new(1, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        let expected = roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]);
        assert!(coordinator.begin(0, 1, 0, 1, expected.clone()).is_err());
        coordinator.begin(0, 2, 32, 1, expected.clone()).unwrap();
        assert!(coordinator.begin(0, 4, 33, 1, expected.clone()).is_err());
        assert!(coordinator
            .begin(QWEN35_LAYER_COUNT, 4, 1, 1, expected.clone())
            .is_err());
        assert!(coordinator.begin(0, 5, 1, 0, expected.clone()).is_err());
        coordinator.begin(0, 3, 1, 1, expected.clone()).unwrap();
        assert!(coordinator.begin(0, 3, 1, 1, expected).is_err());
        assert_eq!(coordinator.state(3).unwrap(), LayerState::Aborted);
    }

    #[test]
    fn any_wrong_stream_or_duplicate_role_aborts_the_transaction() {
        let coordinator = LayerCoordinator::new(5, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        begin_three_role_layer(&coordinator, 200);
        assert!(coordinator
            .capture_projection(5, 200, 0xdead, projection(12, Iq1sExpertRole::Gate, 2, 5))
            .is_err());
        assert_eq!(coordinator.state(200).unwrap(), LayerState::Aborted);

        begin_three_role_layer(&coordinator, 201);
        coordinator
            .capture_projection(5, 201, 0xabc0, projection(12, Iq1sExpertRole::Gate, 2, 5))
            .unwrap();
        assert!(coordinator
            .capture_projection(5, 201, 0xabc0, projection(12, Iq1sExpertRole::Gate, 2, 5))
            .is_err());
        assert_eq!(coordinator.state(201).unwrap(), LayerState::Aborted);
    }

    #[test]
    fn invalid_expert_missing_phase_a_and_early_down_fail_closed() {
        let coordinator = LayerCoordinator::new(6, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        coordinator
            .begin(
                12,
                300,
                1,
                0xabc0,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        let mut bad_routes = routes(1);
        bad_routes[0].expert_id = QWEN35_EXPERT_COUNT as u16;
        assert!(coordinator.set_routes_host(300, bad_routes).is_err());
        assert_eq!(coordinator.state(300).unwrap(), LayerState::Aborted);

        begin_three_role_layer(&coordinator, 301);
        assert!(coordinator.commit_phase_a(301).is_err());
        assert_eq!(coordinator.state(301).unwrap(), LayerState::Aborted);

        begin_three_role_layer(&coordinator, 302);
        assert!(coordinator
            .capture_projection(6, 302, 0xabc0, projection(12, Iq1sExpertRole::Down, 2, 6))
            .is_err());
        assert_eq!(coordinator.state(302).unwrap(), LayerState::Aborted);

        coordinator
            .begin(
                12,
                303,
                1,
                0xabc0,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        unsafe {
            assert!(coordinator
                .set_routes_device(
                    303,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    QWEN35_EXPERTS_PER_TOKEN - 1,
                )
                .is_err());
        }
        assert_eq!(coordinator.state(303).unwrap(), LayerState::Aborted);

        coordinator
            .begin(
                12,
                304,
                1,
                0xabc0,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        coordinator.set_routes_host(304, routes(1)).unwrap();
        assert!(coordinator.set_routes_host(304, routes(1)).is_err());
        assert_eq!(coordinator.state(304).unwrap(), LayerState::Aborted);

        coordinator
            .begin(
                12,
                305,
                1,
                0xabc0,
                roles(&[Iq1sExpertRole::Gate, Iq1sExpertRole::Up]),
            )
            .unwrap();
        let mut duplicate_expert = routes(1);
        duplicate_expert[1].expert_id = duplicate_expert[0].expert_id;
        assert!(coordinator.set_routes_host(305, duplicate_expert).is_err());
        assert_eq!(coordinator.state(305).unwrap(), LayerState::Aborted);
    }

    #[test]
    fn stale_session_or_allocation_generation_aborts() {
        let coordinator = LayerCoordinator::new(11, QWEN35_EXPERTS_PER_TOKEN).unwrap();
        begin_three_role_layer(&coordinator, 400);
        assert!(coordinator
            .capture_projection(10, 400, 0xabc0, projection(12, Iq1sExpertRole::Gate, 2, 11))
            .is_err());
        assert_eq!(coordinator.state(400).unwrap(), LayerState::Aborted);

        begin_three_role_layer(&coordinator, 401);
        let mut stale = projection(12, Iq1sExpertRole::Gate, 2, 10);
        stale.launches[0].launch.allocation_generation = 11;
        assert!(coordinator
            .capture_projection(11, 401, 0xabc0, stale)
            .is_err());
        assert_eq!(coordinator.state(401).unwrap(), LayerState::Aborted);
    }

    #[test]
    fn exported_v2_ffi_rejects_wrong_abi_before_dereferencing_cuda_pointers() {
        unsafe {
            assert_eq!(
                crate::hetgpu_iq1s_layer_begin_v2(1, 0, 990_001, 1, std::ptr::null_mut()),
                HETGPU_IQ1S_ERROR
            );
            assert_eq!(
                crate::hetgpu_iq1s_layer_set_routes_v2(
                    1,
                    990_001,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    QWEN35_EXPERTS_PER_TOKEN,
                ),
                HETGPU_IQ1S_ERROR
            );
        }
        assert_eq!(
            crate::hetgpu_iq1s_layer_phase_commit_v2(1, 990_001, HETGPU_IQ1S_PHASE_A),
            HETGPU_IQ1S_ERROR
        );
        assert_eq!(
            crate::hetgpu_iq1s_layer_commit_v2(1, 990_001),
            HETGPU_IQ1S_ERROR
        );
        assert_eq!(
            crate::hetgpu_iq1s_layer_abort_v2(1, 990_001, 7),
            HETGPU_IQ1S_ERROR
        );
    }

    #[test]
    fn native_route_dma_requires_the_cuda_batch_copy_and_event_symbols() {
        let api = cuda_route_api().expect("GPU host must provide CUDA 13 batch-DMA symbols");
        assert_ne!(api.memcpy_batch_async as usize, 0);
        assert_ne!(api.event_record as usize, 0);
        assert_ne!(api.event_synchronize as usize, 0);
    }

    #[test]
    fn cuda13_route_dma_attributes_are_stream_ordered() {
        let attributes = cuda_stream_ordered_copy_attributes();
        assert_eq!(std::mem::size_of::<CuMemcpyAttributes>(), 24);
        assert_eq!(attributes.src_access_order, 1);
        assert_eq!(attributes.src_loc_hint.location_type, 0);
        assert_eq!(attributes.src_loc_hint.id, 0);
        assert_eq!(attributes.dst_loc_hint.location_type, 0);
        assert_eq!(attributes.dst_loc_hint.id, 0);
        assert_eq!(attributes.flags, 0);
    }
}
