use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::mem::size_of;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const V3_VERSION: u32 = 3;
pub(crate) const CAP_DAX_RANGES: u64 = 1 << 0;
pub(crate) const CAP_DIRECT_DESCRIPTOR: u64 = 1 << 6;
pub(crate) const REQUIRED_HARDWARE_CAPABILITIES: u64 = 0x7a;
pub(crate) const CAP_DAX_FD_BIND_REQUIRED: u64 = 1 << 7;
pub(crate) const REQUIRED_PRE_BIND: u64 = REQUIRED_HARDWARE_CAPABILITIES | CAP_DAX_FD_BIND_REQUIRED;
pub(crate) const REQUIRED_POST_BIND: u64 = REQUIRED_PRE_BIND | CAP_DAX_RANGES;
pub(crate) const LANE_ANY: u32 = u32::MAX;
/// Matches the loaded driver's `TMATMUL_V3_MAX_LANES` wave limit.
pub(crate) const MAX_LANES: u32 = 4;
pub(crate) const BUFFER_TERNARY2: u32 = 1;
pub(crate) const BUFFER_Q8_8_S16: u32 = 2;
pub(crate) const BUFFER_RAW_S64: u32 = 3;
pub(crate) const BUFFER_READ: u32 = 1 << 0;
pub(crate) const BUFFER_WRITE: u32 = 1 << 1;
pub(crate) const BUFFER_MATRIX: u32 = 1 << 2;

const EXPECTED_DIM_D: u32 = 2048;
const EXPECTED_DDR_BITS: u32 = 512;
const EXPECTED_DAX_HPA: u64 = 0x0c10_0000_0000;
const EXPECTED_DAX_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const REQUIRED_LANE_MASK: u64 = (1_u64 << MAX_LANES) - 1;
const EXPECTED_CLOCK_HZ: u64 = 400_000_000;
const DEFAULT_TIMEOUT_MS: u32 = 5_000;
const MAX_PROOF_TIMEOUT_MS: u32 = 10_000;
const MAX_SANE_QUEUE_DEPTH: u32 = 1 << 20;

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

const fn ioc<T>(direction: u32, magic: u8, number: u8) -> u64 {
    ((direction as u64) << IOC_DIRSHIFT)
        | ((magic as u64) << IOC_TYPESHIFT)
        | ((number as u64) << IOC_NRSHIFT)
        | ((size_of::<T>() as u64) << IOC_SIZESHIFT)
}

pub(crate) const fn iow<T>(magic: u8, number: u8) -> u64 {
    ioc::<T>(IOC_WRITE, magic, number)
}

pub(crate) const fn iowr<T>(magic: u8, number: u8) -> u64 {
    ioc::<T>(IOC_READ | IOC_WRITE, magic, number)
}

pub(crate) const QUERY_CAPS_V3: u64 = 0xC080_CE10;
pub(crate) const REGISTER_BUFFER_V3: u64 = 0xC040_CE11;
pub(crate) const UNREGISTER_BUFFER_V3: u64 = 0x4040_CE12;
pub(crate) const COMMIT_BUFFER_V3: u64 = 0xC040_CE13;
pub(crate) const SUBMIT_V3: u64 = 0xC040_CE14;
pub(crate) const WAIT_V3: u64 = 0xC040_CE15;
pub(crate) const BIND_DAX_V3: u64 = 0xC048_CE16;
pub(crate) const UNBIND_DAX_V3: u64 = 0x4040_CE17;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapsV3 {
    pub(crate) size: u32,
    pub(crate) version: u32,
    pub(crate) capabilities: u64,
    pub(crate) num_instances: u32,
    pub(crate) dim_d: u32,
    pub(crate) max_batch: u32,
    pub(crate) max_descriptors: u32,
    pub(crate) max_inflight_submissions: u32,
    pub(crate) max_timeout_ms: u32,
    pub(crate) ddr_data_width_bits: u32,
    pub(crate) dax_alignment_bytes: u32,
    pub(crate) dax_bytes: u64,
    pub(crate) per_lane_counter_mask: u64,
    pub(crate) accelerator_clock_hz: u64,
    reserved: [u64; 7],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BindDaxV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) dax_fd: i32,
    reserved0: u32,
    pub(crate) hpa_start: u64,
    pub(crate) dax_bytes: u64,
    pub(crate) generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnbindDaxV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) generation: u64,
    reserved: [u64; 6],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) dpa_offset: u64,
    pub(crate) length: u64,
    pub(crate) format: u32,
    pub(crate) handle: u32,
    pub(crate) generation: u64,
    reserved: [u64; 3],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) handle: u32,
    reserved0: u32,
    pub(crate) expected_generation: u64,
    pub(crate) new_generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TaskV3 {
    pub(crate) request_id: u64,
    pub(crate) flags: u32,
    pub(crate) lane: u32,
    pub(crate) batch: u32,
    pub(crate) valid_out: u32,
    pub(crate) valid_in: u32,
    pub(crate) matrix_handle: u32,
    pub(crate) input_handle: u32,
    pub(crate) output_handle: u32,
    reserved0: u32,
    pub(crate) matrix_generation: u64,
    pub(crate) matrix_offset: u64,
    pub(crate) input_offset: u64,
    pub(crate) input_stride_bytes: u64,
    pub(crate) output_offset: u64,
    pub(crate) output_stride_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubmitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) count: u32,
    reserved0: u32,
    pub(crate) tasks_ptr: u64,
    pub(crate) submission_id: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompletionV3 {
    pub(crate) request_id: u64,
    pub(crate) status: i32,
    pub(crate) lane_used: u32,
    pub(crate) flags: u32,
    reserved0: u32,
    pub(crate) accelerator_cycles: u64,
    pub(crate) matrix_bytes_read: u64,
    pub(crate) input_bytes_read: u64,
    pub(crate) output_bytes_written: u64,
    pub(crate) start_cycle: u64,
    pub(crate) end_cycle: u64,
    reserved: [u64; 1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompletedTaskV3 {
    submission_id: u64,
    task: TaskV3,
    completion: CompletionV3,
}

impl CompletedTaskV3 {
    pub(crate) fn submission_id(&self) -> u64 {
        self.submission_id
    }

    pub(crate) fn task(&self) -> &TaskV3 {
        &self.task
    }

    pub(crate) fn completion(&self) -> &CompletionV3 {
        &self.completion
    }

    /// Builds scheduler fixtures that exercise report-local shape and overflow checks without
    /// exposing an unchecked constructor in production.
    #[cfg(test)]
    pub(crate) fn test_only_new_unchecked(
        submission_id: u64,
        task: TaskV3,
        completion: CompletionV3,
    ) -> Self {
        Self {
            submission_id,
            task,
            completion,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WaitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) timeout_ms: u32,
    pub(crate) max_completions: u32,
    pub(crate) submission_id: u64,
    pub(crate) completions_ptr: u64,
    pub(crate) completed: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

const _: [(); 128] = [(); size_of::<CapsV3>()];
const _: [(); 72] = [(); size_of::<BindDaxV3>()];
const _: [(); 64] = [(); size_of::<UnbindDaxV3>()];
const _: [(); 64] = [(); size_of::<BufferV3>()];
const _: [(); 64] = [(); size_of::<CommitV3>()];
const _: [(); 128] = [(); size_of::<TaskV3>()];
const _: [(); 64] = [(); size_of::<SubmitV3>()];
const _: [(); 80] = [(); size_of::<CompletionV3>()];
const _: [(); 64] = [(); size_of::<WaitV3>()];

pub(crate) trait IoctlOps {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String>;
    fn bind_dax(&mut self, _bind: &mut BindDaxV3) -> Result<(), String> {
        Err("BIND_DAX is not implemented by this test backend".into())
    }
    fn unbind_dax(&mut self, _unbind: &mut UnbindDaxV3) -> Result<(), String> {
        Err("UNBIND_DAX is not implemented by this test backend".into())
    }
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String>;
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String>;
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String>;
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String>;
    fn wait(&mut self, wait: &mut WaitV3, completions: &mut [CompletionV3]) -> Result<(), String>;
}

impl<T: IoctlOps + ?Sized> IoctlOps for &mut T {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
        (**self).query_caps(caps)
    }
    fn bind_dax(&mut self, bind: &mut BindDaxV3) -> Result<(), String> {
        (**self).bind_dax(bind)
    }
    fn unbind_dax(&mut self, unbind: &mut UnbindDaxV3) -> Result<(), String> {
        (**self).unbind_dax(unbind)
    }
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        (**self).register_buffer(buffer)
    }
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        (**self).unregister_buffer(buffer)
    }
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
        (**self).commit_buffer(commit)
    }
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
        (**self).submit(submit)
    }
    fn wait(&mut self, wait: &mut WaitV3, completions: &mut [CompletionV3]) -> Result<(), String> {
        (**self).wait(wait, completions)
    }
}

#[derive(Debug)]
pub(crate) struct LinuxIoctl {
    file: File,
}

impl LinuxIoctl {
    fn open(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        Ok(Self { file })
    }

    fn call<T>(&mut self, request: u64, value: &mut T) -> Result<(), String> {
        let result = unsafe {
            libc::ioctl(
                self.file.as_raw_fd(),
                request as libc::c_ulong,
                value as *mut T,
            )
        };
        if result < 0 {
            Err(format!(
                "ioctl request=0x{request:x}: {}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(())
        }
    }
}

impl IoctlOps for LinuxIoctl {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
        self.call(QUERY_CAPS_V3, caps)
    }
    fn bind_dax(&mut self, bind: &mut BindDaxV3) -> Result<(), String> {
        self.call(BIND_DAX_V3, bind)
    }
    fn unbind_dax(&mut self, unbind: &mut UnbindDaxV3) -> Result<(), String> {
        self.call(UNBIND_DAX_V3, unbind)
    }
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        self.call(REGISTER_BUFFER_V3, buffer)
    }
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        self.call(UNREGISTER_BUFFER_V3, buffer)
    }
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
        self.call(COMMIT_BUFFER_V3, commit)
    }
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
        self.call(SUBMIT_V3, submit)
    }
    fn wait(&mut self, wait: &mut WaitV3, _completions: &mut [CompletionV3]) -> Result<(), String> {
        self.call(WAIT_V3, wait)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferLease {
    owner: u64,
    handle: u32,
    generation: u64,
}

impl BufferLease {
    pub(crate) fn handle(self) -> u32 {
        self.handle
    }
    pub(crate) fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone)]
struct BufferState {
    wire: BufferV3,
    committed: bool,
    quarantined: bool,
    recyclable: bool,
}

static NEXT_SESSION_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(crate) struct V3Session<I: IoctlOps> {
    io: I,
    caps: CapsV3,
    owner: u64,
    timeout_ms: u32,
    last_submission_id: u64,
    buffers: HashMap<u32, BufferState>,
    used_request_ids: HashSet<u64>,
    poisoned: bool,
    bound_generation: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct BoundDaxFile {
    file: File,
    length: u64,
    generation: u64,
}

impl BoundDaxFile {
    pub(crate) fn length(&self) -> u64 {
        self.length
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn test_only(file: File, length: u64) -> Self {
        Self {
            file,
            length,
            generation: 1,
        }
    }
}

impl AsRawFd for BoundDaxFile {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl V3Session<LinuxIoctl> {
    pub(crate) fn open(
        control_path: impl AsRef<Path>,
        dax_path: impl AsRef<Path>,
    ) -> Result<(Self, BoundDaxFile), String> {
        let io = LinuxIoctl::open(control_path.as_ref())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_SYNC | libc::O_CLOEXEC)
            .open(dax_path.as_ref())
            .map_err(|error| format!("open DAX {}: {error}", dax_path.as_ref().display()))?;
        let session = Self::with_bound_io(io, file.as_raw_fd())?;
        let bound = BoundDaxFile {
            length: session.caps.dax_bytes,
            generation: session
                .bound_generation
                .ok_or("bound session lost its DAX generation")?,
            file,
        };
        Ok((session, bound))
    }
}

impl<I: IoctlOps> V3Session<I> {
    /// Test constructor for a backend whose DAX range is already bound.
    #[cfg(test)]
    pub(crate) fn with_io(mut io: I) -> Result<Self, String> {
        let mut caps = CapsV3 {
            size: size_of::<CapsV3>() as u32,
            ..CapsV3::default()
        };
        io.query_caps(&mut caps)?;
        validate_caps(&caps, CapsPhase::PostBind)?;
        Self::from_validated_caps(io, caps, None)
    }

    pub(crate) fn with_bound_io(mut io: I, dax_fd: RawFd) -> Result<Self, String> {
        let mut pre = CapsV3 {
            size: size_of::<CapsV3>() as u32,
            ..CapsV3::default()
        };
        io.query_caps(&mut pre)?;
        validate_caps(&pre, CapsPhase::PreBind)?;

        let mut bind = BindDaxV3 {
            size: size_of::<BindDaxV3>() as u32,
            dax_fd,
            ..BindDaxV3::default()
        };
        io.bind_dax(&mut bind)?;
        validate_bind_result(&bind)?;

        let mut post = CapsV3 {
            size: size_of::<CapsV3>() as u32,
            ..CapsV3::default()
        };
        io.query_caps(&mut post)?;
        if let Err(error) = validate_caps(&post, CapsPhase::PostBind) {
            let mut unbind = UnbindDaxV3 {
                size: size_of::<UnbindDaxV3>() as u32,
                generation: bind.generation,
                ..UnbindDaxV3::default()
            };
            let cleanup = io.unbind_dax(&mut unbind);
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(format!(
                    "{error}; unbind after failed post-bind validation: {cleanup}"
                )),
            };
        }
        Self::from_validated_caps(io, post, Some(bind.generation))
    }

    fn from_validated_caps(
        io: I,
        caps: CapsV3,
        bound_generation: Option<u64>,
    ) -> Result<Self, String> {
        let timeout_ms = DEFAULT_TIMEOUT_MS.min(caps.max_timeout_ms);
        let owner = NEXT_SESSION_OWNER.fetch_add(1, Ordering::Relaxed);
        if owner == 0 {
            return Err("session owner identifier wrapped".into());
        }
        Ok(Self {
            io,
            caps,
            owner,
            timeout_ms,
            last_submission_id: 0,
            buffers: HashMap::new(),
            used_request_ids: HashSet::new(),
            poisoned: false,
            bound_generation,
        })
    }

    pub(crate) fn caps(&self) -> CapsV3 {
        self.caps
    }

    #[cfg(test)]
    pub(crate) fn io(&self) -> &I {
        &self.io
    }

    #[cfg(test)]
    pub(crate) fn io_mut(&mut self) -> &mut I {
        &mut self.io
    }

    pub(crate) fn set_timeout_ms(&mut self, timeout_ms: u32) -> Result<(), String> {
        self.ensure_healthy()?;
        if timeout_ms == 0 || timeout_ms > self.caps.max_timeout_ms {
            return Err(format!(
                "timeout_ms {timeout_ms} outside 1..={}",
                self.caps.max_timeout_ms
            ));
        }
        self.timeout_ms = timeout_ms;
        Ok(())
    }

    pub(crate) fn register_buffer(
        &mut self,
        dpa_offset: u64,
        length: u64,
        format: u32,
        flags: u32,
    ) -> Result<BufferLease, String> {
        self.ensure_healthy()?;
        let alignment = u64::from(self.caps.dax_alignment_bytes);
        if length == 0 || !dpa_offset.is_multiple_of(alignment) || !length.is_multiple_of(alignment)
        {
            return Err(format!(
                "buffer range must be nonempty and {alignment}-byte aligned"
            ));
        }
        let end = dpa_offset
            .checked_add(length)
            .ok_or_else(|| "buffer range overflow".to_string())?;
        if end > self.caps.dax_bytes {
            return Err(format!("buffer range ends outside DAX at {end}"));
        }
        if self.buffers.values().any(|state| {
            let other_start = state.wire.dpa_offset;
            let other_end = other_start + state.wire.length;
            dpa_offset < other_end && other_start < end
        }) {
            return Err("buffer range overlaps an existing registration".into());
        }
        if !matches!(format, BUFFER_TERNARY2 | BUFFER_Q8_8_S16 | BUFFER_RAW_S64) {
            return Err(format!("unsupported buffer format {format}"));
        }
        let valid_flags = BUFFER_READ | BUFFER_WRITE | BUFFER_MATRIX;
        if flags == 0
            || flags & !valid_flags != 0
            || flags & BUFFER_MATRIX != 0 && flags & BUFFER_READ == 0
        {
            return Err(format!("invalid buffer flags 0x{flags:x}"));
        }
        let mut wire = BufferV3 {
            size: size_of::<BufferV3>() as u32,
            flags,
            dpa_offset,
            length,
            format,
            ..BufferV3::default()
        };
        self.io.register_buffer(&mut wire)?;
        if self.buffers.contains_key(&wire.handle) {
            self.poisoned = true;
            return Err(format!(
                "REGISTER_BUFFER returned duplicate live handle {}; session poisoned",
                wire.handle
            ));
        }
        if wire.handle == 0 {
            self.poisoned = true;
            return Err("REGISTER_BUFFER returned invalid zero handle; session poisoned".into());
        }
        let lease = BufferLease {
            owner: self.owner,
            handle: wire.handle,
            generation: wire.generation,
        };
        self.buffers.insert(
            wire.handle,
            BufferState {
                wire,
                committed: false,
                quarantined: false,
                recyclable: false,
            },
        );
        Ok(lease)
    }

    pub(crate) fn commit_buffer(&mut self, lease: BufferLease) -> Result<BufferLease, String> {
        self.ensure_healthy()?;
        if lease.owner != self.owner {
            return Err("buffer lease belongs to a different session owner".into());
        }
        let state = self
            .buffers
            .get(&lease.handle)
            .ok_or_else(|| format!("unknown buffer handle {}", lease.handle))?;
        if state.wire.generation != lease.generation {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                if state.wire.flags & BUFFER_MATRIX != 0 {
                    state.quarantined = true;
                    state.recyclable = false;
                }
            }
            return Err(format!(
                "buffer handle {} generation mismatch",
                lease.handle
            ));
        }
        if state.quarantined {
            return Err(format!("handle {} is quarantined", lease.handle));
        }
        if state.wire.flags & BUFFER_MATRIX == 0 {
            return Err(format!("handle {} is not a matrix buffer", lease.handle));
        }
        if state.committed {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                state.quarantined = true;
                state.recyclable = false;
            }
            return Err(format!(
                "matrix handle {} is already committed and is now quarantined",
                lease.handle
            ));
        }
        let mut commit = CommitV3 {
            size: size_of::<CommitV3>() as u32,
            handle: lease.handle,
            expected_generation: lease.generation,
            ..CommitV3::default()
        };
        if let Err(error) = self.io.commit_buffer(&mut commit) {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                state.quarantined = true;
                state.recyclable = false;
            }
            return Err(format!("COMMIT_BUFFER attempted: {error}"));
        }
        if commit.new_generation <= lease.generation {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                state.quarantined = true;
                state.recyclable = false;
            }
            return Err(format!(
                "COMMIT_BUFFER returned non-increasing generation {}",
                commit.new_generation
            ));
        }
        let state = self
            .buffers
            .get_mut(&lease.handle)
            .ok_or_else(|| format!("buffer handle {} disappeared during commit", lease.handle))?;
        state.wire.generation = commit.new_generation;
        state.committed = true;
        Ok(BufferLease {
            generation: commit.new_generation,
            ..lease
        })
    }

    pub(crate) fn unregister_buffer(&mut self, lease: BufferLease) -> Result<(), String> {
        self.ensure_healthy()?;
        let state = self.checked_state(lease)?;
        if state.quarantined {
            return Err(format!("handle {} is quarantined", lease.handle));
        }
        let mut wire = state.wire;
        if let Err(error) = self.io.unregister_buffer(&mut wire) {
            self.poisoned = true;
            return Err(format!(
                "UNREGISTER_BUFFER for handle {} failed ambiguously; session poisoned: {error}",
                lease.handle
            ));
        }
        self.buffers.remove(&lease.handle);
        Ok(())
    }

    pub(crate) fn is_quarantined(&self, lease: BufferLease) -> bool {
        self.checked_state(lease)
            .map(|state| state.quarantined)
            .unwrap_or(false)
    }

    pub(crate) fn is_recyclable(&self, lease: BufferLease) -> bool {
        self.checked_state(lease)
            .map(|state| state.recyclable && !state.quarantined)
            .unwrap_or(false)
    }

    pub(crate) fn run_tasks(&mut self, tasks: &[TaskV3]) -> Result<Vec<CompletedTaskV3>, String> {
        self.ensure_healthy()?;
        if let Err(error) = self.validate_tasks(tasks) {
            for handle in referenced_handles(tasks) {
                if let Some(state) = self.buffers.get_mut(&handle) {
                    if state.wire.flags & BUFFER_MATRIX == 0 && !state.quarantined {
                        state.recyclable = true;
                    }
                }
            }
            return Err(error);
        }
        let mut all = HashMap::with_capacity(tasks.len());
        let max_descriptors = self.caps.max_descriptors as usize;
        for batch in tasks.chunks(max_descriptors) {
            let handles = referenced_handles(batch);
            let batch_result = self.run_batch(batch);
            match batch_result {
                Ok(completions) => {
                    for completion in completions {
                        all.insert(completion.task.request_id, completion);
                    }
                    for handle in handles {
                        if let Some(state) = self.buffers.get_mut(&handle) {
                            if state.wire.flags & BUFFER_MATRIX == 0 {
                                state.recyclable = true;
                            }
                        }
                    }
                }
                Err(error) => {
                    for handle in handles {
                        if let Some(state) = self.buffers.get_mut(&handle) {
                            state.quarantined = true;
                            state.recyclable = false;
                        }
                    }
                    return Err(error);
                }
            }
        }
        tasks
            .iter()
            .map(|task| {
                all.remove(&task.request_id)
                    .ok_or_else(|| format!("missing completion request_id={}", task.request_id))
            })
            .collect()
    }

    fn run_batch(&mut self, tasks: &[TaskV3]) -> Result<Vec<CompletedTaskV3>, String> {
        let mut submit = SubmitV3 {
            size: size_of::<SubmitV3>() as u32,
            count: tasks.len() as u32,
            tasks_ptr: tasks.as_ptr() as usize as u64,
            ..SubmitV3::default()
        };
        for task in tasks {
            if !self.used_request_ids.insert(task.request_id) {
                return Err(format!(
                    "request_id={} was already used by this session",
                    task.request_id
                ));
            }
        }
        self.io
            .submit(&mut submit)
            .map_err(|error| format!("SUBMIT attempted: {error}"))?;
        if submit.submission_id == 0 || submit.submission_id <= self.last_submission_id {
            return Err(format!(
                "invalid non-increasing submission_id={} after {}",
                submit.submission_id, self.last_submission_id
            ));
        }
        self.last_submission_id = submit.submission_id;

        let expected: HashMap<u64, &TaskV3> =
            tasks.iter().map(|task| (task.request_id, task)).collect();
        let mut found = HashMap::with_capacity(tasks.len());
        while found.len() < tasks.len() {
            let remaining = tasks.len() - found.len();
            let mut wire_completions = vec![CompletionV3::default(); remaining];
            let mut wait = WaitV3 {
                size: size_of::<WaitV3>() as u32,
                timeout_ms: self.timeout_ms,
                max_completions: remaining as u32,
                submission_id: submit.submission_id,
                completions_ptr: wire_completions.as_mut_ptr() as usize as u64,
                ..WaitV3::default()
            };
            self.io
                .wait(&mut wait, &mut wire_completions)
                .map_err(|error| format!("WAIT submission_id={}: {error}", submit.submission_id))?;
            let completed = wait.completed as usize;
            if completed == 0 {
                let mut missing: Vec<u64> = expected
                    .keys()
                    .filter(|request_id| !found.contains_key(*request_id))
                    .copied()
                    .collect();
                missing.sort_unstable();
                return Err(format!(
                    "WAIT zero progress for submission_id={}; missing request_ids={missing:?}",
                    submit.submission_id,
                ));
            }
            if completed > wire_completions.len() {
                return Err(format!(
                    "WAIT completed count {completed} exceeds requested {remaining}"
                ));
            }
            for completion in wire_completions.into_iter().take(completed) {
                let task = expected.get(&completion.request_id).ok_or_else(|| {
                    format!("unknown completion request_id={}", completion.request_id)
                })?;
                if found.contains_key(&completion.request_id) {
                    return Err(format!(
                        "duplicate completion request_id={}",
                        completion.request_id
                    ));
                }
                self.validate_completion(task, &completion)?;
                found.insert(completion.request_id, completion);
            }
        }
        tasks
            .iter()
            .map(|task| {
                found
                    .remove(&task.request_id)
                    .map(|completion| CompletedTaskV3 {
                        submission_id: submit.submission_id,
                        task: *task,
                        completion,
                    })
                    .ok_or_else(|| format!("missing completion request_id={}", task.request_id))
            })
            .collect()
    }

    fn validate_tasks(&self, tasks: &[TaskV3]) -> Result<(), String> {
        if tasks.is_empty() {
            return Err("task list is empty".into());
        }
        validate_output_nonoverlap(tasks, self.caps.dim_d)?;
        let mut request_ids = HashSet::with_capacity(tasks.len());
        for task in tasks {
            if task.request_id == 0 {
                return Err("request_id must be nonzero".into());
            }
            if !request_ids.insert(task.request_id) {
                return Err(format!("duplicate request_id={}", task.request_id));
            }
            if self.used_request_ids.contains(&task.request_id) {
                return Err(format!(
                    "request_id={} was reused in this open session",
                    task.request_id
                ));
            }
            if task.flags != 0 || task.reserved0 != 0 || task.reserved != [0; 4] {
                return Err(format!(
                    "request_id={} has nonzero flags/reserved",
                    task.request_id
                ));
            }
            if task.lane != LANE_ANY
                && (task.lane >= MAX_LANES
                    || task.lane >= self.caps.num_instances
                    || self.caps.per_lane_counter_mask & (1_u64 << task.lane) == 0)
            {
                return Err(format!(
                    "request_id={} has invalid lane {}",
                    task.request_id, task.lane
                ));
            }
            if task.batch == 0 || task.batch > self.caps.max_batch {
                return Err(format!("request_id={} has invalid batch", task.request_id));
            }
            if task.valid_out == 0
                || task.valid_out > self.caps.dim_d
                || task.valid_in == 0
                || task.valid_in > self.caps.dim_d
            {
                return Err(format!(
                    "request_id={} has invalid geometry",
                    task.request_id
                ));
            }
            let matrix = self.state_for_handle(task.matrix_handle, "matrix")?;
            let input = self.state_for_handle(task.input_handle, "input")?;
            let output = self.state_for_handle(task.output_handle, "output")?;
            if task.matrix_handle == task.input_handle
                || task.matrix_handle == task.output_handle
                || task.input_handle == task.output_handle
            {
                return Err(format!(
                    "request_id={} must reference three distinct handles",
                    task.request_id
                ));
            }
            if matrix.quarantined || input.quarantined || output.quarantined {
                return Err(format!(
                    "request_id={} references quarantined lease",
                    task.request_id
                ));
            }
            if matrix.wire.flags & (BUFFER_MATRIX | BUFFER_READ) != BUFFER_MATRIX | BUFFER_READ {
                return Err(format!(
                    "request_id={} matrix handle has wrong flags",
                    task.request_id
                ));
            }
            if matrix.wire.format != BUFFER_TERNARY2
                || input.wire.format != BUFFER_Q8_8_S16
                || output.wire.format != BUFFER_RAW_S64
            {
                return Err(format!(
                    "request_id={} buffer formats do not match matrix/input/output roles",
                    task.request_id
                ));
            }
            if input.wire.flags & BUFFER_READ == 0 || output.wire.flags & BUFFER_WRITE == 0 {
                return Err(format!(
                    "request_id={} input/output handle has wrong flags",
                    task.request_id
                ));
            }
            if !matrix.committed {
                return Err(format!(
                    "request_id={} matrix handle {} is not committed",
                    task.request_id, task.matrix_handle
                ));
            }
            if task.matrix_generation != matrix.wire.generation {
                return Err(format!(
                    "request_id={} matrix generation mismatch",
                    task.request_id
                ));
            }
            let (matrix_bytes, input_row_bytes, output_row_bytes) =
                task_physical_bytes(self.caps.dim_d)?;
            let stride_alignment = u64::from(self.caps.dax_alignment_bytes);
            let matrix_end = task
                .matrix_offset
                .checked_add(matrix_bytes)
                .ok_or_else(|| "matrix range overflow".to_string())?;
            if matrix_end > matrix.wire.length {
                return Err("matrix range exceeds registered buffer".into());
            }
            if task.input_stride_bytes < input_row_bytes
                || !task.input_stride_bytes.is_multiple_of(stride_alignment)
            {
                return Err(format!(
                    "input stride {} is smaller than {input_row_bytes} or not {stride_alignment}-byte DAX aligned",
                    task.input_stride_bytes,
                ));
            }
            if !task.input_offset.is_multiple_of(2) {
                return Err(format!(
                    "input offset {} is not 2-byte aligned",
                    task.input_offset
                ));
            }
            if task.output_stride_bytes < output_row_bytes
                || !task.output_stride_bytes.is_multiple_of(stride_alignment)
            {
                return Err(format!(
                    "output stride {} is smaller than {output_row_bytes} or not {stride_alignment}-byte DAX aligned",
                    task.output_stride_bytes,
                ));
            }
            if !task.output_offset.is_multiple_of(8) {
                return Err(format!(
                    "output offset {} is not 8-byte aligned",
                    task.output_offset
                ));
            }
            let (_, input_end) = checked_row_extent(
                task.input_offset,
                task.input_stride_bytes,
                task.batch,
                input_row_bytes,
                "input",
            )?;
            if input_end > input.wire.length {
                return Err("input range exceeds registered buffer".into());
            }
            let (_, output_end) = checked_row_extent(
                task.output_offset,
                task.output_stride_bytes,
                task.batch,
                output_row_bytes,
                "output",
            )?;
            if output_end > output.wire.length {
                return Err("output range exceeds registered buffer".into());
            }
        }
        Ok(())
    }

    fn validate_completion(&self, task: &TaskV3, completion: &CompletionV3) -> Result<(), String> {
        if completion.status != 0 {
            return Err(format!(
                "request_id={} completion status={}",
                completion.request_id, completion.status
            ));
        }
        if completion.lane_used >= MAX_LANES
            || completion.lane_used >= self.caps.num_instances
            || self.caps.per_lane_counter_mask & (1_u64 << completion.lane_used) == 0
        {
            return Err(format!(
                "request_id={} invalid completion lane",
                completion.request_id
            ));
        }
        if task.lane != LANE_ANY && completion.lane_used != task.lane {
            return Err(format!(
                "request_id={} completion lane {} does not match requested lane {}",
                completion.request_id, completion.lane_used, task.lane
            ));
        }
        if completion.flags != 0 || completion.reserved0 != 0 || completion.reserved != [0; 1] {
            return Err(format!(
                "request_id={} invalid completion flags/reserved",
                completion.request_id
            ));
        }
        if completion.end_cycle <= completion.start_cycle
            || completion.accelerator_cycles != completion.end_cycle - completion.start_cycle
        {
            return Err(format!(
                "request_id={} invalid completion cycle range",
                completion.request_id
            ));
        }
        let matrix = &self.buffers[&task.matrix_handle].wire;
        let input = &self.buffers[&task.input_handle].wire;
        let output = &self.buffers[&task.output_handle].wire;
        let (minimum_matrix, minimum_input, minimum_output) =
            completion_minimum_bytes(self.caps.dim_d, task.batch)?;
        if completion.matrix_bytes_read < minimum_matrix {
            return Err(format!(
                "request_id={} invalid matrix_bytes_read counter {}, expected at least {minimum_matrix}",
                completion.request_id, completion.matrix_bytes_read
            ));
        }
        if completion.input_bytes_read < minimum_input {
            return Err(format!(
                "request_id={} invalid input_bytes_read counter {}, expected at least {minimum_input}",
                completion.request_id, completion.input_bytes_read
            ));
        }
        if completion.output_bytes_written < minimum_output {
            return Err(format!(
                "request_id={} invalid output_bytes_written counter {}, expected at least {minimum_output}",
                completion.request_id, completion.output_bytes_written
            ));
        }
        let matrix_traffic_limit = matrix
            .length
            .checked_sub(task.matrix_offset)
            .and_then(|bytes| bytes.checked_mul(u64::from(task.batch)))
            .ok_or_else(|| {
                format!(
                    "request_id={} matrix traffic range overflow",
                    completion.request_id
                )
            })?;
        if completion.matrix_bytes_read > matrix_traffic_limit {
            return Err(format!(
                "request_id={} matrix_bytes_read out of range",
                completion.request_id
            ));
        }
        if completion.input_bytes_read > input.length - task.input_offset {
            return Err(format!(
                "request_id={} input_bytes_read out of range",
                completion.request_id
            ));
        }
        if completion.output_bytes_written > output.length - task.output_offset {
            return Err(format!(
                "request_id={} output_bytes_written out of range",
                completion.request_id
            ));
        }
        Ok(())
    }

    fn checked_state(&self, lease: BufferLease) -> Result<&BufferState, String> {
        if lease.owner != self.owner {
            return Err("buffer lease belongs to a different session owner".into());
        }
        let state = self
            .buffers
            .get(&lease.handle)
            .ok_or_else(|| format!("unknown buffer handle {}", lease.handle))?;
        if state.wire.generation != lease.generation {
            return Err(format!(
                "buffer handle {} generation mismatch",
                lease.handle
            ));
        }
        Ok(state)
    }

    fn state_for_handle(&self, handle: u32, role: &str) -> Result<&BufferState, String> {
        if handle == 0 {
            return Err(format!("{role} handle is zero"));
        }
        self.buffers
            .get(&handle)
            .ok_or_else(|| format!("unknown {role} handle {handle}"))
    }

    fn ensure_healthy(&self) -> Result<(), String> {
        if self.poisoned {
            Err("TernIP V3 session is poisoned until fd Drop".into())
        } else {
            Ok(())
        }
    }
}

impl<I: IoctlOps> Drop for V3Session<I> {
    fn drop(&mut self) {
        let Some(generation) = self.bound_generation.take() else {
            return;
        };
        let mut unbind = UnbindDaxV3 {
            size: size_of::<UnbindDaxV3>() as u32,
            generation,
            ..UnbindDaxV3::default()
        };
        if let Err(error) = self.io.unbind_dax(&mut unbind) {
            eprintln!("[CXL TMatmul] UNBIND_DAX generation={generation} failed: {error}");
        }
    }
}

fn completion_minimum_bytes(dim_d: u32, batch: u32) -> Result<(u64, u64, u64), String> {
    let dim_d = u64::from(dim_d);
    let batch = u64::from(batch);
    let matrix = batch
        .checked_mul(dim_d)
        .and_then(|bytes| bytes.checked_mul(dim_d))
        .and_then(|bytes| bytes.checked_div(4))
        .ok_or_else(|| "matrix completion byte minimum overflow".to_string())?;
    let input = batch
        .checked_mul(dim_d)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| "input completion byte minimum overflow".to_string())?;
    let output = batch
        .checked_mul(dim_d)
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or_else(|| "output completion byte minimum overflow".to_string())?;
    Ok((matrix, input, output))
}

#[derive(Debug, Clone, Copy)]
enum CapsPhase {
    PreBind,
    PostBind,
}

fn validate_caps(caps: &CapsV3, phase: CapsPhase) -> Result<(), String> {
    if caps.size != size_of::<CapsV3>() as u32 {
        return Err(format!("caps size {} is not 128", caps.size));
    }
    if caps.version != V3_VERSION {
        return Err(format!("caps version {} is not 3", caps.version));
    }
    let (phase_name, expected_capabilities, expected_dax_bytes) = match phase {
        CapsPhase::PreBind => ("pre-bind", REQUIRED_PRE_BIND, 0),
        CapsPhase::PostBind => ("post-bind", REQUIRED_POST_BIND, EXPECTED_DAX_BYTES),
    };
    if caps.capabilities != expected_capabilities {
        return Err(format!(
            "{phase_name} capabilities 0x{:x} are not exactly 0x{expected_capabilities:x}",
            caps.capabilities,
        ));
    }
    if caps.num_instances != MAX_LANES {
        return Err(format!(
            "caps num_instances {} is not exactly {MAX_LANES}",
            caps.num_instances
        ));
    }
    if caps.dim_d != EXPECTED_DIM_D {
        return Err(format!("caps dim_d {} is not 2048", caps.dim_d));
    }
    if caps.max_batch == 0 {
        return Err("caps max_batch is zero".into());
    }
    if caps.max_descriptors == 0 || caps.max_descriptors > MAX_SANE_QUEUE_DEPTH {
        return Err(format!(
            "caps max_descriptors {} is not sane",
            caps.max_descriptors
        ));
    }
    if caps.max_inflight_submissions == 0 || caps.max_inflight_submissions > MAX_SANE_QUEUE_DEPTH {
        return Err(format!(
            "caps max_inflight {} is not sane",
            caps.max_inflight_submissions
        ));
    }
    if caps.max_timeout_ms == 0 || caps.max_timeout_ms > MAX_PROOF_TIMEOUT_MS {
        return Err(format!(
            "caps max_timeout {} is outside 1..={MAX_PROOF_TIMEOUT_MS}",
            caps.max_timeout_ms
        ));
    }
    if caps.ddr_data_width_bits != EXPECTED_DDR_BITS {
        return Err(format!(
            "caps ddr width {} is not 512",
            caps.ddr_data_width_bits
        ));
    }
    if caps.dax_alignment_bytes == 0 || !caps.dax_alignment_bytes.is_power_of_two() {
        return Err(format!(
            "caps alignment {} is invalid",
            caps.dax_alignment_bytes
        ));
    }
    if caps.dax_bytes != expected_dax_bytes {
        return Err(format!(
            "{phase_name} DAX bytes {} are not {}",
            caps.dax_bytes, expected_dax_bytes
        ));
    }
    if caps.per_lane_counter_mask != REQUIRED_LANE_MASK {
        return Err(format!(
            "caps lane_mask 0x{:x} is not exactly 0x{REQUIRED_LANE_MASK:x}",
            caps.per_lane_counter_mask,
        ));
    }
    if caps.accelerator_clock_hz != EXPECTED_CLOCK_HZ {
        return Err(format!(
            "caps clock {} is not 400 MHz",
            caps.accelerator_clock_hz
        ));
    }
    if caps.reserved != [0; 7] {
        return Err("caps reserved fields are nonzero".into());
    }
    Ok(())
}

fn validate_bind_result(bind: &BindDaxV3) -> Result<(), String> {
    if bind.hpa_start != EXPECTED_DAX_HPA {
        return Err(format!(
            "bound HPA 0x{:x} is not 0x{EXPECTED_DAX_HPA:x}",
            bind.hpa_start
        ));
    }
    if bind.dax_bytes != EXPECTED_DAX_BYTES {
        return Err(format!(
            "bound size 0x{:x} is not 0x{EXPECTED_DAX_BYTES:x}",
            bind.dax_bytes
        ));
    }
    if bind.generation == 0 {
        return Err("bound DAX generation is zero".into());
    }
    Ok(())
}

fn task_physical_bytes(dim_d: u32) -> Result<(u64, u64, u64), String> {
    let dim_d = u64::from(dim_d);
    let matrix = dim_d
        .checked_mul(dim_d)
        .map(|bytes| bytes / 4)
        .ok_or_else(|| "matrix range overflow".to_string())?;
    let input = dim_d
        .checked_mul(2)
        .ok_or_else(|| "input range overflow".to_string())?;
    let output = dim_d
        .checked_mul(8)
        .ok_or_else(|| "output range overflow".to_string())?;
    Ok((matrix, input, output))
}

fn checked_row_extent(
    offset: u64,
    stride: u64,
    batch: u32,
    row_bytes: u64,
    role: &str,
) -> Result<(u64, u64), String> {
    if batch == 0 {
        return Err(format!("{role} range has zero batch"));
    }
    let last_row = stride
        .checked_mul(u64::from(batch - 1))
        .and_then(|delta| offset.checked_add(delta))
        .ok_or_else(|| format!("{role} range overflow"))?;
    let end = last_row
        .checked_add(row_bytes)
        .ok_or_else(|| format!("{role} range overflow"))?;
    Ok((offset, end))
}

fn validate_output_nonoverlap(tasks: &[TaskV3], dim_d: u32) -> Result<(), String> {
    let (_, _, output_row_bytes) = task_physical_bytes(dim_d)?;
    let mut by_handle: HashMap<u32, Vec<(u64, u64, u64)>> = HashMap::new();
    for task in tasks {
        let (start, end) = checked_row_extent(
            task.output_offset,
            task.output_stride_bytes,
            task.batch,
            output_row_bytes,
            "output",
        )?;
        by_handle
            .entry(task.output_handle)
            .or_default()
            .push((start, end, task.request_id));
    }
    for ranges in by_handle.values_mut() {
        ranges.sort_unstable_by_key(|range| range.0);
        for adjacent in ranges.windows(2) {
            if adjacent[1].0 < adjacent[0].1 {
                return Err(format!(
                    "output ranges overlap for request_id={} and request_id={}",
                    adjacent[0].2, adjacent[1].2
                ));
            }
        }
    }
    Ok(())
}

fn referenced_handles(tasks: &[TaskV3]) -> HashSet<u32> {
    let mut handles = HashSet::new();
    for task in tasks {
        handles.insert(task.matrix_handle);
        handles.insert(task.input_handle);
        handles.insert(task.output_handle);
    }
    handles
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::mem::size_of;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    type CapsMutation = fn(&mut CapsV3);
    type TaskMutation = fn(&mut TaskV3);

    #[derive(Debug, Default)]
    struct FakeIoctl {
        caps: CapsV3,
        post_bind_caps: Option<CapsV3>,
        bind_hpa: u64,
        bind_bytes: u64,
        bind_generation: u64,
        calls: Vec<String>,
        next_handle: u32,
        next_generation: u64,
        next_submission: u64,
        register_results: VecDeque<(u32, u64)>,
        waits: VecDeque<Result<Vec<CompletionV3>, String>>,
        submit_error: Option<String>,
        commit_error: Option<String>,
        commit_generation: Option<u64>,
        unregister_error: Option<String>,
        wait_timeout_seen: Vec<u32>,
        drops: Option<Arc<AtomicUsize>>,
    }

    impl Drop for FakeIoctl {
        fn drop(&mut self) {
            if let Some(drops) = &self.drops {
                drops.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl FakeIoctl {
        fn valid() -> Self {
            Self {
                caps: valid_caps(),
                post_bind_caps: None,
                bind_hpa: EXPECTED_DAX_HPA,
                bind_bytes: EXPECTED_DAX_BYTES,
                bind_generation: 19,
                calls: Vec::new(),
                next_handle: 1,
                next_generation: 1,
                next_submission: 1,
                register_results: VecDeque::new(),
                waits: VecDeque::new(),
                submit_error: None,
                commit_error: None,
                commit_generation: None,
                unregister_error: None,
                wait_timeout_seen: Vec::new(),
                drops: None,
            }
        }

        fn pre_bind() -> Self {
            let mut io = Self::valid();
            io.caps.capabilities = REQUIRED_PRE_BIND;
            io.caps.dax_bytes = 0;
            io.post_bind_caps = Some(valid_caps());
            io
        }

        fn with_waits(waits: Vec<Vec<CompletionV3>>) -> Self {
            let mut io = Self::valid();
            io.waits = waits.into_iter().map(Ok).collect();
            io
        }
    }

    impl IoctlOps for FakeIoctl {
        fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
            self.calls.push("query".into());
            *caps = self.caps;
            Ok(())
        }

        fn bind_dax(&mut self, bind: &mut BindDaxV3) -> Result<(), String> {
            self.calls.push(format!("bind:{}", bind.dax_fd));
            bind.hpa_start = self.bind_hpa;
            bind.dax_bytes = self.bind_bytes;
            bind.generation = self.bind_generation;
            if let Some(caps) = self.post_bind_caps.take() {
                self.caps = caps;
            }
            Ok(())
        }

        fn unbind_dax(&mut self, unbind: &mut UnbindDaxV3) -> Result<(), String> {
            self.calls.push(format!("unbind:{}", unbind.generation));
            Ok(())
        }

        fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            self.calls.push(format!("register:{}", buffer.dpa_offset));
            if let Some((handle, generation)) = self.register_results.pop_front() {
                buffer.handle = handle;
                buffer.generation = generation;
            } else {
                buffer.handle = self.next_handle;
                buffer.generation = self.next_generation;
                self.next_handle += 1;
            }
            Ok(())
        }

        fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            self.calls.push(format!("unregister:{}", buffer.handle));
            if let Some(error) = self.unregister_error.take() {
                return Err(error);
            }
            Ok(())
        }

        fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
            self.calls.push(format!("commit:{}", commit.handle));
            if let Some(error) = self.commit_error.take() {
                return Err(error);
            }
            commit.new_generation = self
                .commit_generation
                .unwrap_or(commit.expected_generation + 1);
            Ok(())
        }

        fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
            self.calls.push(format!("submit:{}", submit.count));
            if let Some(error) = self.submit_error.take() {
                return Err(error);
            }
            submit.submission_id = self.next_submission;
            self.next_submission += 1;
            Ok(())
        }

        fn wait(
            &mut self,
            wait: &mut WaitV3,
            completions: &mut [CompletionV3],
        ) -> Result<(), String> {
            self.calls.push(format!("wait:{}", wait.submission_id));
            self.wait_timeout_seen.push(wait.timeout_ms);
            let result = self.waits.pop_front().unwrap_or_else(|| Ok(Vec::new()))?;
            assert!(result.len() <= completions.len());
            completions[..result.len()].copy_from_slice(&result);
            wait.completed = result.len() as u32;
            Ok(())
        }
    }

    fn valid_caps() -> CapsV3 {
        CapsV3 {
            size: size_of::<CapsV3>() as u32,
            version: V3_VERSION,
            capabilities: REQUIRED_POST_BIND,
            num_instances: MAX_LANES,
            dim_d: 2048,
            max_batch: 64,
            max_descriptors: 2,
            max_inflight_submissions: 4,
            max_timeout_ms: 5000,
            ddr_data_width_bits: 512,
            dax_alignment_bytes: 4096,
            dax_bytes: 32 * 1024 * 1024 * 1024,
            per_lane_counter_mask: REQUIRED_LANE_MASK,
            accelerator_clock_hz: 400_000_000,
            ..CapsV3::default()
        }
    }

    fn completion(request_id: u64) -> CompletionV3 {
        CompletionV3 {
            request_id,
            lane_used: 3,
            accelerator_cycles: 10,
            start_cycle: 20,
            end_cycle: 30,
            matrix_bytes_read: 2048 * 2048 / 4,
            input_bytes_read: 4096,
            output_bytes_written: 16384,
            ..CompletionV3::default()
        }
    }

    fn register_triplet<I: IoctlOps>(session: &mut V3Session<I>) -> [BufferLease; 3] {
        register_triplet_at(session, 0)
    }

    fn register_triplet_at<I: IoctlOps>(session: &mut V3Session<I>, base: u64) -> [BufferLease; 3] {
        let mut leases = [
            session
                .register_buffer(
                    base,
                    2048 * 2048 / 4,
                    BUFFER_TERNARY2,
                    BUFFER_READ | BUFFER_MATRIX,
                )
                .unwrap(),
            session
                .register_buffer(base + 2048 * 2048 / 4, 8192, BUFFER_Q8_8_S16, BUFFER_READ)
                .unwrap(),
            session
                .register_buffer(
                    base + 2048 * 2048 / 4 + 8192,
                    65536,
                    BUFFER_RAW_S64,
                    BUFFER_WRITE,
                )
                .unwrap(),
        ];
        leases[0] = session.commit_buffer(leases[0]).unwrap();
        leases
    }

    fn task(request_id: u64, leases: &[BufferLease; 3]) -> TaskV3 {
        TaskV3 {
            request_id,
            lane: LANE_ANY,
            batch: 1,
            valid_out: 1,
            valid_in: 1,
            matrix_handle: leases[0].handle(),
            input_handle: leases[1].handle(),
            output_handle: leases[2].handle(),
            matrix_generation: leases[0].generation(),
            input_stride_bytes: 4096,
            output_stride_bytes: 16384,
            ..TaskV3::default()
        }
    }

    fn disjoint_tasks(request_ids: &[u64], leases: &[BufferLease; 3]) -> Vec<TaskV3> {
        request_ids
            .iter()
            .enumerate()
            .map(|(index, request_id)| {
                let mut task = task(*request_id, leases);
                task.output_offset = index as u64 * 16384;
                task
            })
            .collect()
    }

    #[test]
    fn v3_uapi_layout_and_ioctl_numbers_match_loaded_driver() {
        assert_eq!(MAX_LANES, 4);
        assert_eq!(size_of::<CapsV3>(), 128);
        assert_eq!(size_of::<BufferV3>(), 64);
        assert_eq!(size_of::<CommitV3>(), 64);
        assert_eq!(size_of::<TaskV3>(), 128);
        assert_eq!(size_of::<SubmitV3>(), 64);
        assert_eq!(size_of::<CompletionV3>(), 80);
        assert_eq!(size_of::<WaitV3>(), 64);
        assert_eq!(size_of::<BindDaxV3>(), 72);
        assert_eq!(size_of::<UnbindDaxV3>(), 64);
        assert_eq!(QUERY_CAPS_V3, 0xC080_CE10);
        assert_eq!(REGISTER_BUFFER_V3, 0xC040_CE11);
        assert_eq!(UNREGISTER_BUFFER_V3, 0x4040_CE12);
        assert_eq!(COMMIT_BUFFER_V3, 0xC040_CE13);
        assert_eq!(SUBMIT_V3, 0xC040_CE14);
        assert_eq!(WAIT_V3, 0xC040_CE15);
        assert_eq!(BIND_DAX_V3, 0xC048_CE16);
        assert_eq!(UNBIND_DAX_V3, 0x4040_CE17);
        assert_eq!(QUERY_CAPS_V3, iowr::<CapsV3>(0xCE, 0x10));
        assert_eq!(REGISTER_BUFFER_V3, iowr::<BufferV3>(0xCE, 0x11));
        assert_eq!(UNREGISTER_BUFFER_V3, iow::<BufferV3>(0xCE, 0x12));
        assert_eq!(COMMIT_BUFFER_V3, iowr::<CommitV3>(0xCE, 0x13));
        assert_eq!(SUBMIT_V3, iowr::<SubmitV3>(0xCE, 0x14));
        assert_eq!(WAIT_V3, iowr::<WaitV3>(0xCE, 0x15));
        assert_eq!(BIND_DAX_V3, iowr::<BindDaxV3>(0xCE, 0x16));
        assert_eq!(UNBIND_DAX_V3, iow::<UnbindDaxV3>(0xCE, 0x17));
    }

    #[test]
    fn devdax_binding_is_pre_bind_bind_post_bind_in_that_order() {
        let session =
            V3Session::with_bound_io(FakeIoctl::pre_bind(), 17).expect("bind one devdax fd");
        assert_eq!(session.caps().capabilities, REQUIRED_POST_BIND);
        assert_eq!(session.caps().dax_bytes, EXPECTED_DAX_BYTES);
        assert_eq!(session.io().calls, ["query", "bind:17", "query"]);
    }

    #[test]
    fn devdax_binding_fails_closed_at_each_phase() {
        let mut cases = Vec::new();

        let mut missing_bind = FakeIoctl::pre_bind();
        missing_bind.caps.capabilities &= !CAP_DAX_FD_BIND_REQUIRED;
        cases.push(("pre-bind capabilities", missing_bind));

        let mut premature_dax = FakeIoctl::pre_bind();
        premature_dax.caps.capabilities |= CAP_DAX_RANGES;
        cases.push(("pre-bind capabilities", premature_dax));

        let mut premature_size = FakeIoctl::pre_bind();
        premature_size.caps.dax_bytes = EXPECTED_DAX_BYTES;
        cases.push(("pre-bind DAX bytes", premature_size));

        let mut zero_generation = FakeIoctl::pre_bind();
        zero_generation.bind_generation = 0;
        cases.push(("generation", zero_generation));

        let mut wrong_hpa = FakeIoctl::pre_bind();
        wrong_hpa.bind_hpa += 4096;
        cases.push(("HPA", wrong_hpa));

        let mut wrong_size = FakeIoctl::pre_bind();
        wrong_size.bind_bytes /= 2;
        cases.push(("size", wrong_size));

        let mut missing_post_dax = FakeIoctl::pre_bind();
        missing_post_dax
            .post_bind_caps
            .as_mut()
            .unwrap()
            .capabilities &= !CAP_DAX_RANGES;
        cases.push(("post-bind capabilities", missing_post_dax));

        let mut post_regression = FakeIoctl::pre_bind();
        post_regression
            .post_bind_caps
            .as_mut()
            .unwrap()
            .capabilities &= !CAP_DIRECT_DESCRIPTOR;
        cases.push(("post-bind capabilities", post_regression));

        for (label, io) in cases {
            let error = V3Session::with_bound_io(io, 17).unwrap_err();
            assert!(error.contains(label), "{label}: {error}");
        }
    }

    #[test]
    fn caps_fail_closed_one_field_at_a_time() {
        let invalid: [(&str, CapsMutation); 12] = [
            ("version", |c: &mut CapsV3| c.version = 2),
            ("capabilities", |c: &mut CapsV3| c.capabilities = 0x3b),
            ("num_instances", |c: &mut CapsV3| c.num_instances = 3),
            ("dim_d", |c: &mut CapsV3| c.dim_d = 1024),
            ("max_descriptors", |c: &mut CapsV3| c.max_descriptors = 0),
            ("max_inflight", |c: &mut CapsV3| {
                c.max_inflight_submissions = 0
            }),
            ("max_timeout", |c: &mut CapsV3| c.max_timeout_ms = 0),
            ("ddr", |c: &mut CapsV3| c.ddr_data_width_bits = 256),
            ("alignment", |c: &mut CapsV3| c.dax_alignment_bytes = 0),
            ("DAX", |c: &mut CapsV3| c.dax_bytes -= 1),
            ("lane_mask", |c: &mut CapsV3| c.per_lane_counter_mask = 0x7),
            ("clock", |c: &mut CapsV3| {
                c.accelerator_clock_hz = 399_000_000
            }),
        ];
        for (label, mutate) in invalid {
            let mut io = FakeIoctl::valid();
            mutate(&mut io.caps);
            assert!(
                V3Session::with_io(io).unwrap_err().contains(label),
                "{label}"
            );
        }

        let mut io = FakeIoctl::valid();
        io.caps.max_descriptors = u32::MAX;
        assert!(V3Session::with_io(io)
            .unwrap_err()
            .contains("max_descriptors"));
    }

    #[test]
    fn caps_accept_exactly_four_runtime_lanes() {
        let mut io = FakeIoctl::valid();
        io.caps.num_instances = MAX_LANES;
        io.caps.per_lane_counter_mask = (1_u64 << MAX_LANES) - 1;

        let session = V3Session::with_io(io).unwrap();

        assert_eq!(session.caps().num_instances, MAX_LANES);
        assert_eq!(session.caps().per_lane_counter_mask, 0xf);
    }

    #[test]
    fn range_registration_rejects_alignment_bounds_and_overlap() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        assert!(session
            .register_buffer(1, 4096, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
        assert!(session
            .register_buffer(0, 4095, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
        assert!(session
            .register_buffer(
                32 * 1024 * 1024 * 1024 - 4096,
                8192,
                BUFFER_RAW_S64,
                BUFFER_READ
            )
            .is_err());
        session
            .register_buffer(0, 8192, BUFFER_RAW_S64, BUFFER_READ)
            .unwrap();
        assert!(session
            .register_buffer(4096, 4096, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
    }

    #[test]
    fn commit_validates_owner_handle_and_generation() {
        let mut first = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let lease = first
            .register_buffer(0, 4096, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
            .unwrap();
        let committed = first.commit_buffer(lease).unwrap();
        assert_eq!(committed.generation(), lease.generation() + 1);
        assert!(first
            .commit_buffer(lease)
            .unwrap_err()
            .contains("generation"));

        let mut second = V3Session::with_io(FakeIoctl::valid()).unwrap();
        assert!(second
            .commit_buffer(committed)
            .unwrap_err()
            .contains("owner"));
    }

    #[test]
    fn ambiguous_or_invalid_commit_quarantines_the_matrix_lease() {
        for invalid_generation in [false, true] {
            let mut io = FakeIoctl::valid();
            if invalid_generation {
                io.commit_generation = Some(1);
            } else {
                io.commit_error = Some("commit EIO".into());
            }
            let mut session = V3Session::with_io(io).unwrap();
            let lease = session
                .register_buffer(0, 4096, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
                .unwrap();
            assert!(session.commit_buffer(lease).is_err());
            assert!(session.is_quarantined(lease));
            assert!(session
                .unregister_buffer(lease)
                .unwrap_err()
                .contains("quarantined"));
        }
    }

    #[test]
    fn uncommitted_or_recommitted_matrix_never_reaches_submit() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let matrix = session
            .register_buffer(
                0,
                2048 * 2048 / 4,
                BUFFER_TERNARY2,
                BUFFER_READ | BUFFER_MATRIX,
            )
            .unwrap();
        let input = session
            .register_buffer(2048 * 2048 / 4, 8192, BUFFER_Q8_8_S16, BUFFER_READ)
            .unwrap();
        let output = session
            .register_buffer(2048 * 2048 / 4 + 8192, 32768, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap();
        let leases = [matrix, input, output];
        assert!(session
            .run_tasks(&[task(1, &leases)])
            .unwrap_err()
            .contains("not committed"));
        assert!(!session
            .io()
            .calls
            .iter()
            .any(|call| call.starts_with("submit")));

        let committed = session.commit_buffer(matrix).unwrap();
        assert!(session
            .commit_buffer(committed)
            .unwrap_err()
            .contains("already committed"));
        assert!(session.is_quarantined(committed));
        assert_eq!(
            session
                .io()
                .calls
                .iter()
                .filter(|call| call.starts_with("commit"))
                .count(),
            1
        );
    }

    #[test]
    fn stale_generation_recommit_quarantines_the_committed_matrix() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let registered = session
            .register_buffer(0, 4096, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
            .unwrap();
        let committed = session.commit_buffer(registered).unwrap();
        assert!(session
            .commit_buffer(registered)
            .unwrap_err()
            .contains("generation"));
        assert!(session.is_quarantined(committed));
    }

    #[test]
    fn partial_wait_accepts_lane_3_and_preserves_input_order() {
        let io = FakeIoctl::with_waits(vec![vec![completion(2)], vec![completion(1)]]);
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let tasks = disjoint_tasks(&[1, 2], &leases);
        let got = session.run_tasks(&tasks).unwrap();
        assert_eq!(
            got.iter()
                .map(|record| record.completion().request_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn completion_lane_must_match_an_explicit_lane_request() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let leases = register_triplet(&mut session);
        let mut explicit = task(1, &leases);
        explicit.lane = 0;
        assert!(session
            .run_tasks(&[explicit])
            .unwrap_err()
            .contains("requested lane"));
    }

    #[test]
    fn lanes_outside_four_lane_runtime_policy_fail_closed() {
        let mut explicit_session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let explicit_leases = register_triplet(&mut explicit_session);
        let mut explicit = task(1, &explicit_leases);
        explicit.lane = MAX_LANES;
        assert!(explicit_session
            .run_tasks(&[explicit])
            .unwrap_err()
            .contains("invalid lane"));

        let mut invalid_completion = completion(1);
        invalid_completion.lane_used = MAX_LANES;
        let mut completion_session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![invalid_completion]])).unwrap();
        let completion_leases = register_triplet(&mut completion_session);
        assert!(completion_session
            .run_tasks(&[task(1, &completion_leases)])
            .unwrap_err()
            .contains("completion lane"));
    }

    #[test]
    fn batches_at_max_descriptors_and_requires_monotonic_submission_ids() {
        let mut io = FakeIoctl::with_waits(vec![
            vec![completion(1), completion(2)],
            vec![completion(3)],
        ]);
        io.next_submission = 40;
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let tasks = disjoint_tasks(&[1, 2, 3], &leases);
        let completed = session.run_tasks(&tasks).unwrap();
        assert_eq!(
            completed
                .iter()
                .map(|record| record.submission_id())
                .collect::<Vec<_>>(),
            vec![40, 40, 41]
        );
        assert_eq!(
            completed
                .iter()
                .map(|record| (record.task().request_id, record.completion().request_id))
                .collect::<Vec<_>>(),
            vec![(1, 1), (2, 2), (3, 3)]
        );
        let copied: CompletedTaskV3 = completed[0];
        assert_eq!(copied.submission_id(), 40);
        assert_eq!(
            session
                .io()
                .calls
                .iter()
                .filter(|c| c.starts_with("submit"))
                .count(),
            2
        );

        session.io_mut().next_submission = 40;
        session.io_mut().waits.push_back(Ok(vec![completion(4)]));
        assert!(session
            .run_tasks(&[task(4, &leases)])
            .unwrap_err()
            .contains("submission_id"));
    }

    #[test]
    fn timeout_is_bounded_to_caps() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        assert!(session.set_timeout_ms(5001).is_err());
        session.set_timeout_ms(5000).unwrap();
        let leases = register_triplet(&mut session);
        session.run_tasks(&[task(1, &leases)]).unwrap();
        assert_eq!(session.io().wait_timeout_seen, vec![5000]);
    }

    #[test]
    fn timeout_caps_enforce_the_ten_second_proof_ceiling() {
        let mut accepted = FakeIoctl::valid();
        accepted.caps.max_timeout_ms = 10_000;
        let mut session = V3Session::with_io(accepted).unwrap();
        session.set_timeout_ms(10_000).unwrap();
        assert!(session.set_timeout_ms(10_001).is_err());

        let mut rejected = FakeIoctl::valid();
        rejected.caps.max_timeout_ms = 10_001;
        assert!(V3Session::with_io(rejected)
            .unwrap_err()
            .contains("max_timeout"));
    }

    #[test]
    fn wait_rejects_zero_progress_duplicate_unknown_and_bad_completions() {
        let cases = [
            (vec![vec![]], "zero progress"),
            (
                vec![vec![completion(1), completion(1)]],
                "duplicate completion request_id=1",
            ),
            (
                vec![vec![completion(99)]],
                "unknown completion request_id=99",
            ),
        ];
        for (waits, expected) in cases {
            let mut session = V3Session::with_io(FakeIoctl::with_waits(waits)).unwrap();
            let leases = register_triplet(&mut session);
            let tasks = disjoint_tasks(&[1, 2], &leases);
            assert!(session.run_tasks(&tasks).unwrap_err().contains(expected));
        }

        for (bad, expected) in [
            (
                {
                    let mut c = completion(1);
                    c.status = -5;
                    c
                },
                "status",
            ),
            (
                {
                    let mut c = completion(1);
                    c.lane_used = 16;
                    c
                },
                "lane",
            ),
            (
                {
                    let mut c = completion(1);
                    c.flags = 1;
                    c
                },
                "flags",
            ),
            (
                {
                    let mut c = completion(1);
                    c.end_cycle = 19;
                    c
                },
                "cycle",
            ),
            (
                {
                    let mut c = completion(1);
                    c.output_bytes_written = 9000;
                    c
                },
                "output_bytes",
            ),
        ] {
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![bad]])).unwrap();
            let leases = register_triplet(&mut session);
            assert!(
                session
                    .run_tasks(&[task(1, &leases)])
                    .unwrap_err()
                    .contains(expected),
                "{expected}: {bad:?}"
            );
        }
    }

    #[test]
    fn proof_counters_require_exact_positive_cycles_and_minimum_traffic() {
        let cycle_cases = [
            {
                let mut c = completion(1);
                c.end_cycle = c.start_cycle;
                c.accelerator_cycles = 0;
                c
            },
            {
                let mut c = completion(1);
                c.accelerator_cycles = 9;
                c
            },
            {
                let mut c = completion(1);
                c.accelerator_cycles = 11;
                c
            },
        ];
        for done in cycle_cases {
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
            let leases = register_triplet(&mut session);
            assert!(session
                .run_tasks(&[task(1, &leases)])
                .unwrap_err()
                .contains("cycle"));
        }

        for (counter, value) in [
            ("matrix", 0),
            ("matrix", 2048 * 2048 / 4 - 1),
            ("input", 0),
            ("input", 2048 * 2 - 1),
            ("output", 0),
            ("output", 2048 * 8 - 1),
        ] {
            let mut done = completion(1);
            match counter {
                "matrix" => done.matrix_bytes_read = value,
                "input" => done.input_bytes_read = value,
                "output" => done.output_bytes_written = value,
                _ => unreachable!(),
            }
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
            let leases = register_triplet(&mut session);
            let error = session.run_tasks(&[task(1, &leases)]).unwrap_err();
            assert!(error.contains(counter), "{counter}={value}: {error}");
        }
    }

    #[test]
    fn proof_counter_minimums_scale_with_batch() {
        for counter in ["matrix", "input", "output"] {
            let mut done = completion(1);
            done.matrix_bytes_read = 2 * 2048 * 2048 / 4;
            done.input_bytes_read = 2 * 2048 * 2;
            done.output_bytes_written = 2 * 2048 * 8;
            match counter {
                "matrix" => done.matrix_bytes_read -= 1,
                "input" => done.input_bytes_read -= 1,
                "output" => done.output_bytes_written -= 1,
                _ => unreachable!(),
            }
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
            let leases = register_triplet(&mut session);
            let mut batched = task(1, &leases);
            batched.batch = 2;
            assert!(session.run_tasks(&[batched]).unwrap_err().contains(counter));
        }

        let mut done = completion(2);
        done.matrix_bytes_read = 2 * 2048 * 2048 / 4;
        done.input_bytes_read = 2 * 2048 * 2;
        done.output_bytes_written = 2 * 2048 * 8;
        let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
        let leases = register_triplet(&mut session);
        let mut batched = task(2, &leases);
        batched.batch = 2;
        session.run_tasks(&[batched]).unwrap();

        assert!(completion_minimum_bytes(u32::MAX, u32::MAX)
            .unwrap_err()
            .contains("overflow"));
    }

    #[test]
    fn completion_byte_ranges_are_relative_to_task_offsets() {
        for counter in ["matrix", "input", "output"] {
            let mut done = completion(1);
            match counter {
                "matrix" => done.matrix_bytes_read = 2048 * 2048 / 4 + 1,
                "input" => done.input_bytes_read = 8192 + 1,
                "output" => done.output_bytes_written = 65536 - 4096 + 1,
                _ => unreachable!(),
            }
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
            let leases = register_triplet(&mut session);
            let mut offset = task(1, &leases);
            if counter == "output" {
                offset.output_offset = 4096;
            }
            assert!(session.run_tasks(&[offset]).unwrap_err().contains(counter));
        }
    }

    #[test]
    fn missing_completion_is_rejected_when_driver_overreports_count() {
        let mut io = FakeIoctl::valid();
        io.waits
            .push_back(Err("timeout with missing request_id=2".into()));
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let tasks = disjoint_tasks(&[1, 2], &leases);
        assert!(session.run_tasks(&tasks).unwrap_err().contains("missing"));
    }

    #[test]
    fn zero_progress_names_request_ids_still_missing_after_partial_wait() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)], vec![]])).unwrap();
        let leases = register_triplet(&mut session);
        let tasks = disjoint_tasks(&[1, 2], &leases);
        let error = session.run_tasks(&tasks).unwrap_err();
        assert!(error.contains("missing request_ids=[2]"), "{error}");
    }

    #[test]
    fn task_requires_distinct_handles_with_exact_role_formats() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let leases = register_triplet(&mut session);
        let mut aliased = task(1, &leases);
        aliased.input_handle = aliased.matrix_handle;
        assert!(session
            .run_tasks(&[aliased])
            .unwrap_err()
            .contains("distinct"));

        let mut other = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let wrong_matrix = other
            .register_buffer(0, 8192, BUFFER_Q8_8_S16, BUFFER_READ | BUFFER_MATRIX)
            .unwrap();
        let input = other
            .register_buffer(8192, 8192, BUFFER_Q8_8_S16, BUFFER_READ)
            .unwrap();
        let output = other
            .register_buffer(16384, 8192, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap();
        let wrong = task(2, &[wrong_matrix, input, output]);
        assert!(other.run_tasks(&[wrong]).unwrap_err().contains("formats"));
    }

    #[test]
    fn full_row_ranges_reject_bad_strides_alignment_overrun_and_overflow() {
        let mut cases = Vec::new();
        let placeholder = [BufferLease {
            owner: 0,
            handle: 0,
            generation: 0,
        }; 3];
        let base = task(1, &placeholder);
        let mutations: [(&str, TaskMutation); 16] = [
            ("input stride", |task: &mut TaskV3| {
                task.input_stride_bytes = 1
            }),
            ("output stride", |task: &mut TaskV3| {
                task.output_stride_bytes = 1
            }),
            ("input stride", |task: &mut TaskV3| {
                task.input_stride_bytes = 4094
            }),
            ("output stride", |task: &mut TaskV3| {
                task.output_stride_bytes = 16376
            }),
            ("input stride", |task: &mut TaskV3| {
                task.input_stride_bytes = 4097
            }),
            ("output stride", |task: &mut TaskV3| {
                task.output_stride_bytes = 16385
            }),
            ("input stride", |task: &mut TaskV3| {
                task.input_stride_bytes = 4098
            }),
            ("output stride", |task: &mut TaskV3| {
                task.output_stride_bytes = 16392
            }),
            ("input offset", |task: &mut TaskV3| task.input_offset = 1),
            ("output offset", |task: &mut TaskV3| task.output_offset = 1),
            ("input range", |task: &mut TaskV3| task.input_offset = 2),
            ("output range", |task: &mut TaskV3| {
                task.output_offset = 32_776
            }),
            ("input range", |task: &mut TaskV3| {
                task.input_offset = u64::MAX - 1
            }),
            ("output range", |task: &mut TaskV3| {
                task.output_offset = u64::MAX - 7
            }),
            ("matrix range", |task: &mut TaskV3| task.matrix_offset = 1),
            ("matrix range", |task: &mut TaskV3| {
                task.matrix_offset = u64::MAX
            }),
        ];
        for (label, mutate) in mutations {
            let mut candidate = base;
            candidate.batch = 2;
            mutate(&mut candidate);
            cases.push((label, candidate));
        }

        for (label, mut candidate) in cases {
            let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
            let leases = register_triplet(&mut session);
            candidate.matrix_handle = leases[0].handle();
            candidate.input_handle = leases[1].handle();
            candidate.output_handle = leases[2].handle();
            candidate.matrix_generation = leases[0].generation();
            let error = session.run_tasks(&[candidate]).unwrap_err();
            assert!(error.contains(label), "{label}: {error}");
            assert!(!session
                .io()
                .calls
                .iter()
                .any(|call| call.starts_with("submit")));
        }
    }

    #[test]
    fn task_strides_follow_the_caps_dax_alignment() {
        let mut live =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let live_leases = register_triplet(&mut live);
        let exact_rows = task(1, &live_leases);
        assert_eq!(exact_rows.input_stride_bytes, 4096);
        assert_eq!(exact_rows.output_stride_bytes, 16384);
        live.run_tasks(&[exact_rows]).unwrap();

        let mut io = FakeIoctl::with_waits(vec![vec![completion(2)]]);
        io.caps.dax_alignment_bytes = 8192;
        let mut aligned = V3Session::with_io(io).unwrap();
        let aligned_leases = register_triplet(&mut aligned);
        let mut aligned_task = task(2, &aligned_leases);
        aligned_task.input_stride_bytes = 8192;
        aligned.run_tasks(&[aligned_task]).unwrap();

        let mut io = FakeIoctl::valid();
        io.caps.dax_alignment_bytes = 8192;
        let mut misaligned = V3Session::with_io(io).unwrap();
        let misaligned_leases = register_triplet(&mut misaligned);
        let error = misaligned
            .run_tasks(&[task(3, &misaligned_leases)])
            .unwrap_err();
        assert!(error.contains("input stride"), "{error}");
        assert!(!misaligned
            .io()
            .calls
            .iter()
            .any(|call| call.starts_with("submit")));
    }

    #[test]
    fn output_ranges_allow_exact_touching_but_reject_one_byte_overlap() {
        let io = FakeIoctl::with_waits(vec![vec![completion(1), completion(2)]]);
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let touching = disjoint_tasks(&[1, 2], &leases);
        session.run_tasks(&touching).unwrap();

        let mut overlap_session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let overlap_leases = register_triplet(&mut overlap_session);
        let mut overlapping = disjoint_tasks(&[3, 4], &overlap_leases);
        overlapping[1].output_offset -= 1;
        assert!(overlap_session
            .run_tasks(&overlapping)
            .unwrap_err()
            .contains("overlap"));
        assert!(!overlap_session
            .io()
            .calls
            .iter()
            .any(|call| call.starts_with("submit")));
    }

    #[test]
    fn request_ids_are_session_unique_from_each_submit_attempt() {
        let mut success =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let success_leases = register_triplet(&mut success);
        success.run_tasks(&[task(1, &success_leases)]).unwrap();
        assert!(success
            .run_tasks(&[task(1, &success_leases)])
            .unwrap_err()
            .contains("reused"));
        assert_eq!(
            success
                .io()
                .calls
                .iter()
                .filter(|call| call.starts_with("submit"))
                .count(),
            1
        );

        let mut failed_io = FakeIoctl::valid();
        failed_io.submit_error = Some("submit EIO".into());
        let mut failed = V3Session::with_io(failed_io).unwrap();
        let failed_leases = register_triplet(&mut failed);
        let attempted = disjoint_tasks(&[10, 11, 12], &failed_leases);
        assert!(failed.run_tasks(&attempted).is_err());

        let fresh_leases = register_triplet_at(&mut failed, 2 * 1024 * 1024);
        failed.io_mut().waits.push_back(Ok(vec![completion(12)]));
        failed.run_tasks(&[task(12, &fresh_leases)]).unwrap();
        assert!(failed
            .run_tasks(&[task(10, &fresh_leases)])
            .unwrap_err()
            .contains("reused"));
        assert_eq!(
            failed
                .io()
                .calls
                .iter()
                .filter(|call| call.starts_with("submit"))
                .count(),
            2
        );
    }

    #[test]
    fn task_ids_and_handles_are_validated_before_submit_without_quarantine() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let leases = register_triplet(&mut session);
        assert!(session.run_tasks(&[]).is_err());
        assert!(session.run_tasks(&[task(0, &leases)]).is_err());
        assert!(session
            .run_tasks(&[task(1, &leases), task(1, &leases)])
            .is_err());
        let mut bad = task(2, &leases);
        bad.matrix_handle = 999;
        assert!(session.run_tasks(&[bad]).is_err());
        assert!(!session.is_quarantined(leases[1]));
        assert!(session.is_recyclable(leases[1]));
        assert!(session.is_recyclable(leases[2]));
        assert!(!session.is_recyclable(leases[0]));
        assert_eq!(
            session
                .io()
                .calls
                .iter()
                .filter(|c| c.starts_with("submit"))
                .count(),
            0
        );
    }

    #[test]
    fn any_failure_after_submit_attempt_quarantines_all_referenced_leases() {
        let mut io = FakeIoctl::valid();
        io.submit_error = Some("submit EIO".into());
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        assert!(session.run_tasks(&[task(1, &leases)]).is_err());
        for lease in leases {
            assert!(session.is_quarantined(lease));
            assert!(session
                .unregister_buffer(lease)
                .unwrap_err()
                .contains("quarantined"));
        }
        assert!(!session
            .io()
            .calls
            .iter()
            .any(|c| c.starts_with("unregister")));
    }

    #[test]
    fn timeout_after_submit_quarantines_and_session_drop_drops_fd_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut io = FakeIoctl::valid();
            io.drops = Some(drops.clone());
            io.waits.push_back(Err("ETIMEDOUT".into()));
            let mut session = V3Session::with_io(io).unwrap();
            let leases = register_triplet(&mut session);
            assert!(session
                .run_tasks(&[task(1, &leases)])
                .unwrap_err()
                .contains("ETIMEDOUT"));
            assert!(leases
                .into_iter()
                .all(|lease| session.is_quarantined(lease)));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn successful_submission_recycles_transients_but_keeps_matrix_owned() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let leases = register_triplet(&mut session);
        session.run_tasks(&[task(1, &leases)]).unwrap();
        assert!(session.is_recyclable(leases[1]));
        assert!(session.is_recyclable(leases[2]));
        assert!(!session.is_recyclable(leases[0]));
        session.unregister_buffer(leases[1]).unwrap();
        assert!(session.io().calls.iter().any(|c| c == "unregister:2"));
    }

    #[test]
    fn ambiguous_unregister_poisons_and_retains_the_lease_until_drop() {
        let mut io = FakeIoctl::valid();
        io.unregister_error = Some("unregister EIO".into());
        let mut session = V3Session::with_io(io).unwrap();
        let lease = session
            .register_buffer(0, 4096, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap();
        assert!(session.unregister_buffer(lease).is_err());
        assert!(session
            .set_timeout_ms(1000)
            .unwrap_err()
            .contains("poisoned"));
        assert_eq!(
            session.io().calls,
            vec!["query", "register:0", "unregister:1"]
        );
    }

    #[test]
    fn successful_register_with_zero_handle_poisons_the_session() {
        for cleanup_fails in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            {
                let mut io = FakeIoctl::valid();
                io.register_results.push_back((0, 1));
                io.unregister_error = cleanup_fails.then(|| "cleanup EIO".into());
                io.drops = Some(drops.clone());
                let mut poisoned = V3Session::with_io(io).unwrap();
                assert!(poisoned
                    .register_buffer(0, 4096, BUFFER_RAW_S64, BUFFER_WRITE)
                    .is_err());
                assert!(poisoned
                    .set_timeout_ms(1000)
                    .unwrap_err()
                    .contains("poisoned"));
                assert!(poisoned.run_tasks(&[]).unwrap_err().contains("poisoned"));
                assert!(!poisoned
                    .io()
                    .calls
                    .iter()
                    .any(|call| call.starts_with("submit")));
                assert_eq!(poisoned.io().calls, vec!["query", "register:0"]);
                assert_eq!(drops.load(Ordering::SeqCst), 0);
            }
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn register_accepts_driver_initial_generation_zero_and_commit_advances_it() {
        let mut io = FakeIoctl::valid();
        io.register_results.push_back((9, 0));
        let mut session = V3Session::with_io(io).unwrap();
        let registered = session
            .register_buffer(
                0,
                2048 * 2048 / 4,
                BUFFER_TERNARY2,
                BUFFER_READ | BUFFER_MATRIX,
            )
            .unwrap();
        assert_eq!(registered.generation(), 0);

        let committed = session.commit_buffer(registered).unwrap();

        assert_eq!(committed.generation(), 1);
        session.unregister_buffer(committed).unwrap();
        assert_eq!(
            session.io().calls,
            vec!["query", "register:0", "commit:9", "unregister:9"]
        );
    }

    #[test]
    fn duplicate_successful_register_poisons_without_unregistering_live_handle() {
        let mut io = FakeIoctl::valid();
        io.register_results.push_back((5, 1));
        io.register_results.push_back((5, 2));
        let mut session = V3Session::with_io(io).unwrap();
        let live = session
            .register_buffer(0, 4096, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap();
        assert!(session
            .register_buffer(4096, 4096, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap_err()
            .contains("duplicate"));
        assert!(!session.io().calls.iter().any(|call| call == "unregister:5"));
        assert!(session.run_tasks(&[]).unwrap_err().contains("poisoned"));
        assert!(session
            .register_buffer(8192, 4096, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap_err()
            .contains("poisoned"));
        assert!(session
            .commit_buffer(live)
            .unwrap_err()
            .contains("poisoned"));
        assert!(session
            .unregister_buffer(live)
            .unwrap_err()
            .contains("poisoned"));
        assert_eq!(
            session.io().calls,
            vec!["query", "register:0", "register:4096"]
        );
    }

    #[test]
    fn causal_order_is_register_commit_submit_partial_wait_then_unregister() {
        let io = FakeIoctl::with_waits(vec![vec![completion(2)], vec![completion(1)]]);
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let tasks = disjoint_tasks(&[1, 2], &leases);
        session.run_tasks(&tasks).unwrap();
        session.unregister_buffer(leases[1]).unwrap();
        assert_eq!(
            session.io().calls,
            vec![
                "query",
                "register:0",
                "register:1048576",
                "register:1056768",
                "commit:1",
                "submit:2",
                "wait:1",
                "wait:1",
                "unregister:2",
            ]
        );
    }
}
