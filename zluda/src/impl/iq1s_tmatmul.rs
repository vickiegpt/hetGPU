//! Exact IQ1_S/MMQ decomposition used by the fail-closed TernIP V3 route.

use crate::r#impl::batch_scheduler::{BatchSchedulerConfig, SchedulerReport};
use crate::r#impl::cxl_tmatmul::{copy_cuda_to_host, copy_host_to_cuda};
use crate::r#impl::cxl_tmatmul_v3::{
    BoundDaxFile, BufferLease, CompletedTaskV3, IoctlOps, TaskV3, V3Session, BUFFER_MATRIX,
    BUFFER_Q8_8_S16, BUFFER_RAW_S64, BUFFER_READ, BUFFER_TERNARY2, BUFFER_WRITE, LANE_ANY,
    MAX_LANES,
};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::hash::{Hash, Hasher};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub(crate) const IQ1S_BLOCK_BYTES: usize = 50;
pub(crate) const IQ1S_BLOCK_VALUES: usize = 256;
pub(crate) const Q8_1_BLOCK_BYTES: usize = 36;
pub(crate) const Q8_1_MMQ_BYTES: usize = 144;
pub(crate) const TILE_DIM: usize = 2048;
pub(crate) const TILE_PACKED_BYTES: usize = TILE_DIM * TILE_DIM / 4;
pub(crate) const GRID_ENTRIES: usize = 2048;
const GROUP_VALUES: usize = 32;
/// Materialization ceiling for one logical execution. The live `max_descriptors` capability only
/// controls ioctl submission windows; it does not bound the host vector holding all descriptors.
const MAX_EXECUTION_DESCRIPTORS: usize = 1 << 20;
const DEFAULT_LIBGGML: &str =
    "/home/eabban/BitNet/build-cuda128-gcc12/3rdparty/llama.cpp/ggml/src/libggml.so";

pub(crate) type GridTable = [[i8; 8]; GRID_ENTRIES];

fn checked_execution_descriptor_count(
    component_count: usize,
    slice_count: usize,
) -> Result<usize, String> {
    let descriptor_count = component_count
        .checked_mul(slice_count)
        .ok_or("execution descriptor count overflow")?;
    if descriptor_count > MAX_EXECUTION_DESCRIPTORS {
        return Err(format!(
            "execution requires {descriptor_count} descriptors, exceeding software safety limit {MAX_EXECUTION_DESCRIPTORS}"
        ));
    }
    Ok(descriptor_count)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct GgmlType19Signature {
    pub(crate) kernel: String,
    pub(crate) ne00: u64,
    pub(crate) ne01: u64,
    pub(crate) stride01: u64,
    pub(crate) ne10: u64,
    pub(crate) ne11: u64,
    pub(crate) stride11: u64,
    pub(crate) ne0: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Q8MmqLayout {
    q8_record_count_per_k: usize,
    active_batch_count: usize,
    pitch_records: usize,
    storage_bytes: usize,
}

impl Q8MmqLayout {
    fn record_range(
        self,
        k_record_index: usize,
        batch_index: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        if k_record_index >= self.q8_record_count_per_k {
            return Err("Q8 K record index is outside the captured activation".into());
        }
        if batch_index >= self.active_batch_count {
            return Err("Q8 batch index is outside the captured activation".into());
        }
        let record_index = k_record_index
            .checked_mul(self.pitch_records)
            .and_then(|base| base.checked_add(batch_index))
            .ok_or("Q8 record index overflow")?;
        let offset = record_index
            .checked_mul(Q8_1_MMQ_BYTES)
            .ok_or("Q8 byte offset overflow")?;
        let end = offset
            .checked_add(Q8_1_MMQ_BYTES)
            .ok_or("Q8 byte offset overflow")?;
        if end > self.storage_bytes {
            return Err("Q8 record is outside the captured activation".into());
        }
        Ok(offset..end)
    }
}

impl GgmlType19Signature {
    pub(crate) fn validate(&self) -> Result<Self, String> {
        if self.kernel != "mul_mat_q"
            && !(self.kernel.starts_with("_Z9mul_mat_qI")
                && !self.kernel[14..].is_empty()
                && self.kernel[14..]
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'))
        {
            return Err("kernel is not a qualified IQ1_S mul_mat_q symbol".into());
        }
        if [
            self.ne00,
            self.ne01,
            self.stride01,
            self.ne10,
            self.ne11,
            self.stride11,
            self.ne0,
        ]
        .contains(&0)
        {
            return Err("signature dimensions and strides must be positive".into());
        }
        if self.ne00 != self.ne10 || self.ne01 != self.ne0 {
            return Err("ne10/ne0 do not match ne00/ne01".into());
        }
        if !self.ne00.is_multiple_of(IQ1S_BLOCK_VALUES as u64) {
            return Err("ne00 must be divisible by 256".into());
        }
        if self.stride01 < self.ne00 / IQ1S_BLOCK_VALUES as u64 {
            return Err("stride01 is smaller than the IQ1_S block count".into());
        }
        usize::try_from(self.ne00).map_err(|_| "ne00 does not fit usize")?;
        usize::try_from(self.ne01).map_err(|_| "ne01 does not fit usize")?;
        usize::try_from(self.stride01).map_err(|_| "stride01 does not fit usize")?;
        usize::try_from(self.ne10).map_err(|_| "ne10 does not fit usize")?;
        usize::try_from(self.ne11).map_err(|_| "ne11 does not fit usize")?;
        usize::try_from(self.stride11).map_err(|_| "stride11 does not fit usize")?;
        usize::try_from(self.ne0).map_err(|_| "ne0 does not fit usize")?;
        self.q8_mmq_layout()?;
        Ok(self.clone())
    }

    fn q8_mmq_layout(&self) -> Result<Q8MmqLayout, String> {
        if !self.ne10.is_multiple_of(128) {
            return Err("ne10 must be divisible by 128".into());
        }
        let q8_record_count_per_k = self.ne10 / 128;
        if q8_record_count_per_k == 0 {
            return Err("Q8 record count per K must be positive".into());
        }
        if self.stride11 < self.ne11 {
            return Err(format!(
                "stride11 record pitch {} is smaller than active batch {}",
                self.stride11, self.ne11
            ));
        }
        let required_record_count = q8_record_count_per_k
            .checked_sub(1)
            .and_then(|last_k_record| last_k_record.checked_mul(self.stride11))
            .and_then(|prefix| prefix.checked_add(self.ne11))
            .ok_or("activation record extent overflow")?;
        let storage_bytes = required_record_count
            .checked_mul(Q8_1_MMQ_BYTES as u64)
            .ok_or("activation byte extent overflow")?;
        Ok(Q8MmqLayout {
            q8_record_count_per_k: usize::try_from(q8_record_count_per_k)
                .map_err(|_| "Q8 record count per K does not fit usize")?,
            active_batch_count: usize::try_from(self.ne11)
                .map_err(|_| "active batch count does not fit usize")?,
            pitch_records: usize::try_from(self.stride11)
                .map_err(|_| "Q8 record pitch does not fit usize")?,
            storage_bytes: usize::try_from(storage_bytes)
                .map_err(|_| "activation size does not fit usize")?,
        })
    }

    pub(crate) fn matrix_storage_bytes(&self) -> Result<usize, String> {
        self.validate()?;
        let stride = self
            .stride01
            .checked_mul(IQ1S_BLOCK_BYTES as u64)
            .ok_or("matrix stride overflow")?;
        let bytes = self
            .ne01
            .checked_mul(stride)
            .ok_or("matrix size overflow")?;
        usize::try_from(bytes).map_err(|_| "matrix size does not fit usize".into())
    }

    pub(crate) fn activation_storage_bytes(&self) -> Result<usize, String> {
        self.validate()?;
        Ok(self.q8_mmq_layout()?.storage_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct GgmlType19VecSignature {
    pub(crate) kernel: String,
    pub(crate) ncols_x: u64,
    pub(crate) nrows_x: u64,
    pub(crate) nrows_y: u64,
    pub(crate) nrows_dst: u64,
}

impl GgmlType19VecSignature {
    pub(crate) fn validate(&self) -> Result<Self, String> {
        if !self.kernel.to_ascii_lowercase().contains("mul_mat_vec_q")
            || !self.kernel.to_ascii_lowercase().contains("ggml_type19")
        {
            return Err("kernel is not a qualified IQ1_S mul_mat_vec_q symbol".into());
        }
        if [self.ncols_x, self.nrows_x, self.nrows_y, self.nrows_dst].contains(&0) {
            return Err("vector dimensions must be positive".into());
        }
        if self.ncols_x != self.nrows_y {
            return Err("nrows_y does not match the IQ1_S input dimension".into());
        }
        if self.nrows_x != self.nrows_dst {
            return Err("split IQ1_S vector launches are not supported".into());
        }
        if !self.ncols_x.is_multiple_of(IQ1S_BLOCK_VALUES as u64) {
            return Err("ncols_x must be divisible by 256".into());
        }
        if !self.nrows_y.is_multiple_of(128) {
            return Err("nrows_y must be divisible by 128 for the DS4 adapter".into());
        }
        usize::try_from(self.ncols_x).map_err(|_| "ncols_x does not fit usize")?;
        usize::try_from(self.nrows_x).map_err(|_| "nrows_x does not fit usize")?;
        Ok(self.clone())
    }

    pub(crate) fn mmq_signature(&self) -> Result<GgmlType19Signature, String> {
        self.validate()?;
        GgmlType19Signature {
            kernel: "mul_mat_q".to_string(),
            ne00: self.ncols_x,
            ne01: self.nrows_x,
            stride01: self.ncols_x / IQ1S_BLOCK_VALUES as u64,
            ne10: self.nrows_y,
            ne11: 1,
            stride11: 1,
            ne0: self.nrows_dst,
        }
        .validate()
    }

    pub(crate) fn matrix_storage_bytes(&self) -> Result<usize, String> {
        self.mmq_signature()?.matrix_storage_bytes()
    }

    pub(crate) fn activation_storage_bytes(&self) -> Result<usize, String> {
        let blocks = self
            .nrows_y
            .checked_div(32)
            .ok_or("vector activation dimension is too small")?;
        usize::try_from(
            blocks
                .checked_mul(Q8_1_BLOCK_BYTES as u64)
                .ok_or("vector activation size overflow")?,
        )
        .map_err(|_| "vector activation size does not fit usize".into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Iq1sGroup {
    pub(crate) odd_scale: u8,
    pub(crate) delta_sign: i8,
    pub(crate) grid_indices: [usize; 4],
    pub(crate) grid_values: [i8; GROUP_VALUES],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Iq1sBlock {
    pub(crate) d: f32,
    pub(crate) groups: [Iq1sGroup; 8],
}

impl Iq1sBlock {
    pub(crate) fn parse(packed: &[u8], grid: &GridTable) -> Result<Self, String> {
        if packed.len() != IQ1S_BLOCK_BYTES {
            return Err("packed IQ1_S block must be exactly 50 bytes".into());
        }
        let d = half_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
        if !d.is_finite() {
            return Err("IQ1_S d must be finite".into());
        }
        let mut groups = [Iq1sGroup {
            odd_scale: 1,
            delta_sign: 1,
            grid_indices: [0; 4],
            grid_values: [0; GROUP_VALUES],
        }; 8];
        for (group_index, group) in groups.iter_mut().enumerate() {
            let qh_offset = 34 + group_index * 2;
            let qh = u16::from_le_bytes([packed[qh_offset], packed[qh_offset + 1]]);
            group.odd_scale = ((((qh >> 12) & 7) * 2) + 1) as u8;
            group.delta_sign = if qh & 0x8000 == 0 { 1 } else { -1 };
            for position in 0..4 {
                let low = usize::from(packed[2 + group_index * 4 + position]);
                let high = usize::from((qh >> (3 * position)) & 7);
                let index = low | high << 8;
                group.grid_indices[position] = index;
                group.grid_values[position * 8..(position + 1) * 8].copy_from_slice(&grid[index]);
            }
        }
        Ok(Self { d, groups })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Q8_1Block {
    pub(crate) d: f32,
    pub(crate) s: f32,
    pub(crate) qs: [i8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Q8_1MmqBlock {
    pub(crate) subblocks: [Q8_1Block; 4],
}

impl Q8_1MmqBlock {
    pub(crate) fn parse(packed: &[u8]) -> Result<Self, String> {
        if packed.len() != Q8_1_MMQ_BYTES {
            return Err("Q8_1 MMQ DS4 block must be exactly 144 bytes".into());
        }
        let mut subblocks = [Q8_1Block {
            d: 0.0,
            s: 0.0,
            qs: [0; 32],
        }; 4];
        for (index, block) in subblocks.iter_mut().enumerate() {
            block.d = half_to_f32(u16::from_le_bytes([
                packed[index * 4],
                packed[index * 4 + 1],
            ]));
            block.s = half_to_f32(u16::from_le_bytes([
                packed[index * 4 + 2],
                packed[index * 4 + 3],
            ]));
            if !block.d.is_finite() || !block.s.is_finite() {
                return Err(format!("Q8_1 MMQ d/s pair {index} must be finite"));
            }
            for (dst, src) in block
                .qs
                .iter_mut()
                .zip(&packed[16 + index * 32..16 + (index + 1) * 32])
            {
                *dst = *src as i8;
            }
        }
        Ok(Self { subblocks })
    }
}

pub(crate) fn iter_q8_1_mmq<'a>(
    signature: &'a GgmlType19Signature,
    packed: &'a [u8],
) -> Result<impl Iterator<Item = Result<Q8_1MmqBlock, String>> + 'a, String> {
    let expected = signature.activation_storage_bytes()?;
    if packed.len() != expected {
        return Err(format!("Q8_1 MMQ storage must be exactly {expected} bytes"));
    }
    let layout = signature.q8_mmq_layout()?;
    Ok(
        (0..layout.q8_record_count_per_k).flat_map(move |k_record_index| {
            (0..layout.active_batch_count).map(move |batch_index| {
                let range = layout.record_range(k_record_index, batch_index)?;
                let packed = packed
                    .get(range)
                    .ok_or("Q8 record is outside the captured activation")?;
                Q8_1MmqBlock::parse(packed)
            })
        }),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) enum ComponentKind {
    Grid,
    Delta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct TileGeometry {
    pub(crate) row_tile: usize,
    pub(crate) column_tile: usize,
    pub(crate) valid_out: usize,
    pub(crate) valid_in: usize,
    pub(crate) group_count: usize,
}

pub(crate) fn plan_tiles(signature: &GgmlType19Signature) -> Result<Vec<TileGeometry>, String> {
    signature.validate()?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row count overflow")?;
    let columns = usize::try_from(signature.ne00).map_err(|_| "column count overflow")?;
    let mut result = Vec::new();
    for row_start in (0..rows).step_by(TILE_DIM) {
        for column_start in (0..columns).step_by(TILE_DIM) {
            let valid_in = (columns - column_start).min(TILE_DIM);
            result.push(TileGeometry {
                row_tile: row_start / TILE_DIM,
                column_tile: column_start / TILE_DIM,
                valid_out: (rows - row_start).min(TILE_DIM),
                valid_in,
                group_count: valid_in.div_ceil(GROUP_VALUES),
            });
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComponentTile {
    pub(crate) geometry: TileGeometry,
    pub(crate) group32: usize,
    pub(crate) kind: ComponentKind,
    /// Compact row-major valid_out x 32 signed ternary values.
    pub(crate) values: Vec<i8>,
}

impl ComponentTile {
    #[cfg(test)]
    pub(crate) fn from_rows(
        geometry: TileGeometry,
        group32: usize,
        kind: ComponentKind,
        rows: &[Iq1sBlock],
    ) -> Result<Self, String> {
        if geometry.valid_out == 0
            || geometry.valid_out > TILE_DIM
            || geometry.valid_in == 0
            || geometry.valid_in > TILE_DIM
            || rows.len() != geometry.valid_out
            || group32 >= geometry.group_count
        {
            return Err("component tile geometry is invalid".into());
        }
        let global_group = geometry.column_tile * (TILE_DIM / GROUP_VALUES) + group32;
        let group_index = global_group % 8;
        let mut values = Vec::with_capacity(rows.len() * GROUP_VALUES);
        for row in rows {
            let group = row.groups[group_index];
            match kind {
                ComponentKind::Grid => values.extend_from_slice(&group.grid_values),
                ComponentKind::Delta => {
                    values.extend(std::iter::repeat_n(group.delta_sign, GROUP_VALUES))
                }
            }
        }
        Ok(Self {
            geometry,
            group32,
            kind,
            values,
        })
    }

    pub(crate) fn pack_ternary2(&self) -> Result<Vec<u8>, String> {
        if self.values.len() != self.geometry.valid_out * GROUP_VALUES {
            return Err("component payload length does not match valid geometry".into());
        }
        let mut packed = vec![0_u8; TILE_PACKED_BYTES];
        let column_start = self.group32 * GROUP_VALUES;
        for row in 0..self.geometry.valid_out {
            for column in 0..GROUP_VALUES {
                let value = self.values[row * GROUP_VALUES + column];
                let code = match value {
                    -1 => 3,
                    0 => 0,
                    1 => 1,
                    _ => return Err("component is not ternary".into()),
                };
                let element = row * TILE_DIM + column_start + column;
                packed[element / 4] |= code << (2 * (element % 4));
            }
        }
        Ok(packed)
    }
}

#[derive(Debug)]
pub(crate) struct MatrixSource {
    signature: GgmlType19Signature,
    packed: Arc<[u8]>,
    grid: Arc<GridTable>,
    row_stride: usize,
    logical_blocks: usize,
}

impl MatrixSource {
    pub(crate) fn new(
        signature: GgmlType19Signature,
        packed: Arc<[u8]>,
        grid: Arc<GridTable>,
    ) -> Result<Arc<Self>, String> {
        let expected = signature.matrix_storage_bytes()?;
        if packed.len() != expected {
            return Err(format!(
                "packed IQ1_S matrix must be exactly {expected} bytes"
            ));
        }
        let row_stride = usize::try_from(signature.stride01)
            .ok()
            .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
            .ok_or("IQ1_S row stride overflow")?;
        let logical_blocks = usize::try_from(signature.ne00 / IQ1S_BLOCK_VALUES as u64)
            .map_err(|_| "IQ1_S block count overflow")?;
        let logical_bytes = logical_blocks
            .checked_mul(IQ1S_BLOCK_BYTES)
            .ok_or("IQ1_S logical row bytes overflow")?;
        let row_count = usize::try_from(signature.ne01).map_err(|_| "row count overflow")?;
        for row in 0..row_count {
            if packed[row * row_stride + logical_bytes..(row + 1) * row_stride]
                .iter()
                .any(|byte| *byte != 0)
            {
                return Err(format!("IQ1_S row {row} padding must be zero"));
            }
        }
        if std::env::var_os("HETGPU_CXL_TMATMUL_DEBUG_BYTES").is_some() {
            for row in 0..row_count {
                for block in 0..logical_blocks {
                    let offset = row * row_stride + block * IQ1S_BLOCK_BYTES;
                    let raw = u16::from_le_bytes([packed[offset], packed[offset + 1]]);
                    if !half_to_f32(raw).is_finite() {
                        let end = (offset + 8).min(packed.len());
                        return Err(format!(
                            "IQ1_S non-finite block row={row} block={block} offset={offset} raw_d=0x{raw:04x} bytes={:02x?}",
                            &packed[offset..end]
                        ));
                    }
                }
            }
        }
        Ok(Arc::new(Self {
            signature,
            packed,
            grid,
            row_stride,
            logical_blocks,
        }))
    }

    pub(crate) fn component_iter(self: &Arc<Self>) -> Result<ComponentIter, String> {
        let rows = usize::try_from(self.signature.ne0).map_err(|_| "row count overflow")?;
        let columns = usize::try_from(self.signature.ne00).map_err(|_| "column count overflow")?;
        let total = plan_tiles(&self.signature)?
            .iter()
            .map(|tile| tile.group_count * 2)
            .sum();
        Ok(ComponentIter {
            source: self.clone(),
            row_tile: 0,
            column_tile: 0,
            group32: 0,
            kind: ComponentKind::Grid,
            row_tiles: rows.div_ceil(TILE_DIM),
            column_tiles: columns.div_ceil(TILE_DIM),
            generated: 0,
            total,
        })
    }

    fn geometry(&self, row_tile: usize, column_tile: usize) -> Result<TileGeometry, String> {
        let rows = usize::try_from(self.signature.ne0).map_err(|_| "row count overflow")?;
        let columns = usize::try_from(self.signature.ne00).map_err(|_| "column count overflow")?;
        let row_start = row_tile.checked_mul(TILE_DIM).ok_or("row tile overflow")?;
        let column_start = column_tile
            .checked_mul(TILE_DIM)
            .ok_or("column tile overflow")?;
        let valid_in = (columns - column_start).min(TILE_DIM);
        Ok(TileGeometry {
            row_tile,
            column_tile,
            valid_out: (rows - row_start).min(TILE_DIM),
            valid_in,
            group_count: valid_in.div_ceil(GROUP_VALUES),
        })
    }

    pub(crate) fn group(
        &self,
        row: usize,
        global_group: usize,
    ) -> Result<(f32, Iq1sGroup), String> {
        let block_index = global_group / 8;
        let rows = usize::try_from(self.signature.ne0).map_err(|_| "row count overflow")?;
        if block_index >= self.logical_blocks || row >= rows {
            return Err("IQ1_S group coordinate is outside the matrix".into());
        }
        let offset = row
            .checked_mul(self.row_stride)
            .and_then(|base| base.checked_add(block_index * IQ1S_BLOCK_BYTES))
            .ok_or("IQ1_S block offset overflow")?;
        let block = Iq1sBlock::parse(&self.packed[offset..offset + IQ1S_BLOCK_BYTES], &self.grid)?;
        Ok((block.d, block.groups[global_group % 8]))
    }

    fn component(
        &self,
        geometry: TileGeometry,
        group32: usize,
        kind: ComponentKind,
    ) -> Result<ComponentTile, String> {
        let global_group = geometry.column_tile * (TILE_DIM / GROUP_VALUES) + group32;
        let row_start = geometry.row_tile * TILE_DIM;
        let mut values = Vec::with_capacity(geometry.valid_out * GROUP_VALUES);
        for row in row_start..row_start + geometry.valid_out {
            let (_, group) = self.group(row, global_group)?;
            match kind {
                ComponentKind::Grid => values.extend_from_slice(&group.grid_values),
                ComponentKind::Delta => {
                    values.extend(std::iter::repeat_n(group.delta_sign, GROUP_VALUES))
                }
            }
        }
        Ok(ComponentTile {
            geometry,
            group32,
            kind,
            values,
        })
    }
}

pub(crate) struct ComponentIter {
    source: Arc<MatrixSource>,
    row_tile: usize,
    column_tile: usize,
    group32: usize,
    kind: ComponentKind,
    row_tiles: usize,
    column_tiles: usize,
    generated: usize,
    total: usize,
}

impl ComponentIter {
    pub(crate) fn generated_count(&self) -> usize {
        self.generated
    }
    pub(crate) fn remaining_count(&self) -> usize {
        self.total - self.generated
    }
}

impl Iterator for ComponentIter {
    type Item = Result<ComponentTile, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.row_tile >= self.row_tiles {
            return None;
        }
        let geometry = match self.source.geometry(self.row_tile, self.column_tile) {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        let result = self.source.component(geometry, self.group32, self.kind);
        self.generated += 1;
        if self.kind == ComponentKind::Grid {
            self.kind = ComponentKind::Delta;
        } else {
            self.kind = ComponentKind::Grid;
            self.group32 += 1;
            if self.group32 == geometry.group_count {
                self.group32 = 0;
                self.column_tile += 1;
                if self.column_tile == self.column_tiles {
                    self.column_tile = 0;
                    self.row_tile += 1;
                }
            }
        }
        Some(result)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining_count();
        (n, Some(n))
    }
}
impl ExactSizeIterator for ComponentIter {}

#[cfg(test)]
pub(crate) fn decompose_component_tiles(
    signature: &GgmlType19Signature,
    packed_matrix: &[u8],
    grid: &GridTable,
) -> Result<Vec<ComponentTile>, String> {
    let expected = signature.matrix_storage_bytes()?;
    if packed_matrix.len() != expected {
        return Err(format!(
            "packed IQ1_S matrix must be exactly {expected} bytes"
        ));
    }
    let row_stride = usize::try_from(signature.stride01)
        .ok()
        .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
        .ok_or("IQ1_S row stride overflow")?;
    let logical_blocks = usize::try_from(signature.ne00 / IQ1S_BLOCK_VALUES as u64)
        .map_err(|_| "IQ1_S block count overflow")?;
    let logical_bytes = logical_blocks
        .checked_mul(IQ1S_BLOCK_BYTES)
        .ok_or("IQ1_S logical row bytes overflow")?;
    for row in 0..usize::try_from(signature.ne01).map_err(|_| "row count overflow")? {
        let padding = &packed_matrix[row * row_stride + logical_bytes..(row + 1) * row_stride];
        if padding.iter().any(|byte| *byte != 0) {
            return Err(format!("IQ1_S row {row} padding must be zero"));
        }
    }

    let geometries = plan_tiles(signature)?;
    let mut result = Vec::with_capacity(
        geometries
            .iter()
            .map(|geometry| geometry.group_count * 2)
            .sum(),
    );
    for geometry in geometries {
        let row_start = geometry.row_tile * TILE_DIM;
        for group32 in 0..geometry.group_count {
            let global_group = geometry.column_tile * (TILE_DIM / GROUP_VALUES) + group32;
            let block_index = global_group / 8;
            let group_index = global_group % 8;
            if block_index >= logical_blocks {
                return Err("component group exceeds the logical matrix width".into());
            }
            let mut grid_values = Vec::with_capacity(geometry.valid_out * GROUP_VALUES);
            let mut delta_values = Vec::with_capacity(geometry.valid_out * GROUP_VALUES);
            for row in row_start..row_start + geometry.valid_out {
                let offset = row
                    .checked_mul(row_stride)
                    .and_then(|base| base.checked_add(block_index * IQ1S_BLOCK_BYTES))
                    .ok_or("IQ1_S block offset overflow")?;
                let parsed =
                    Iq1sBlock::parse(&packed_matrix[offset..offset + IQ1S_BLOCK_BYTES], grid)?;
                let group = parsed.groups[group_index];
                grid_values.extend_from_slice(&group.grid_values);
                delta_values.extend(std::iter::repeat_n(group.delta_sign, GROUP_VALUES));
            }
            result.push(ComponentTile {
                geometry,
                group32,
                kind: ComponentKind::Grid,
                values: grid_values,
            });
            result.push(ComponentTile {
                geometry,
                group32,
                kind: ComponentKind::Delta,
                values: delta_values,
            });
        }
    }
    Ok(result)
}

pub(crate) fn raw_component_dots(group: &Iq1sGroup, q8: &Q8_1Block) -> (i64, i64) {
    let grid = group
        .grid_values
        .iter()
        .zip(q8.qs)
        .map(|(&w, q)| i64::from(w) * i64::from(q))
        .sum();
    let delta = i64::from(group.delta_sign) * q8.qs.iter().map(|&q| i64::from(q)).sum::<i64>();
    (grid, delta)
}

pub(crate) fn encode_q8_8(qs: &[i8; 32]) -> [i16; 32] {
    qs.map(|quant| i16::from(quant) << 8)
}

pub(crate) fn validate_raw_q8_8(raw: i64, expected: i64) -> Result<i64, String> {
    if raw % 256 != 0 {
        return Err("raw Q8.8 result is not divisible by 256".into());
    }
    let decoded = raw / 256;
    if decoded != expected {
        return Err(format!(
            "raw Q8.8 mismatch: got {decoded}, expected {expected}"
        ));
    }
    Ok(decoded)
}

pub(crate) fn reconstruct_from_raw(
    group: &Iq1sGroup,
    iq1s_d: f32,
    q8: &Q8_1Block,
    grid_raw: i64,
    delta_raw: i64,
) -> Result<f32, String> {
    if !iq1s_d.is_finite() || !q8.d.is_finite() || !q8.s.is_finite() {
        return Err("reconstruction factors must be finite".into());
    }
    let (expected_grid, expected_delta) = raw_component_dots(group, q8);
    let grid_dot = validate_raw_q8_8(grid_raw, expected_grid)?;
    let delta_dot = validate_raw_q8_8(delta_raw, expected_delta)?;
    let d1q = (iq1s_d * f32::from(group.odd_scale)) as f32;
    let wd = round_to_half(d1q)?;
    let delta_factor = (-1.0_f32 + f32::from(group.delta_sign) * 0.125) as f32;
    let wdelta = round_to_half((d1q * delta_factor) as f32)?;
    let unsigned_grid = grid_dot
        .checked_add(i64::from(group.delta_sign) * delta_dot)
        .ok_or("unsigned grid sum overflow")?;
    let scaled_grid = ((wd * q8.d) as f32 * unsigned_grid as f32) as f32;
    let result = (scaled_grid + (wdelta * q8.s) as f32) as f32;
    if !result.is_finite() {
        return Err("reconstructed MMQ output is not finite".into());
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MatrixCacheIdentity {
    pub(crate) matrix_ptr: usize,
    pub(crate) signature: GgmlType19Signature,
    pub(crate) allocation_generation: u64,
    pub(crate) content_hash: [u8; 32],
}

const MAX_COMPONENT_CACHE_ENTRIES: usize = 512;

#[derive(Debug, Default)]
pub(crate) struct ComponentCache {
    sources: HashMap<MatrixCacheIdentity, Weak<MatrixSource>>,
}

impl ComponentCache {
    pub(crate) fn matches(&self, identity: &MatrixCacheIdentity) -> bool {
        self.sources
            .get(identity)
            .is_some_and(|source| source.strong_count() != 0)
    }
    pub(crate) fn get(&self, identity: &MatrixCacheIdentity) -> Option<Arc<MatrixSource>> {
        self.sources.get(identity).and_then(Weak::upgrade)
    }
    pub(crate) fn insert(&mut self, identity: MatrixCacheIdentity, source: Arc<MatrixSource>) {
        self.sources.retain(|_, source| source.strong_count() != 0);
        if self.sources.len() >= MAX_COMPONENT_CACHE_ENTRIES
            && !self.sources.contains_key(&identity)
        {
            return;
        }
        self.sources.insert(identity, Arc::downgrade(&source));
    }
    pub(crate) fn invalidate(&mut self) {
        self.sources.clear();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LogicalLaunch {
    pub(crate) matrix_ptr: usize,
    pub(crate) activation_ptr: usize,
    pub(crate) output_ptr: usize,
    pub(crate) allocation_generation: u64,
    pub(crate) content_hash: [u8; 32],
    pub(crate) signature: GgmlType19Signature,
}

pub(crate) fn checked_output_element_count(
    signature: &GgmlType19Signature,
) -> Result<usize, String> {
    let logical_batch = u32::try_from(signature.ne11).map_err(|_| {
        format!(
            "ne11={} is outside the u32 scheduler batch domain",
            signature.ne11
        )
    })?;
    let logical_batch =
        usize::try_from(logical_batch).map_err(|_| "logical batch does not fit host usize")?;
    let output_rows =
        usize::try_from(signature.ne0).map_err(|_| "output row count does not fit host usize")?;
    let output_elements = logical_batch
        .checked_mul(output_rows)
        .ok_or("output element extent overflow")?;
    output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("output extent overflow")?;
    Ok(output_elements)
}

impl LogicalLaunch {
    pub(crate) fn validate_before_copy(&self) -> Result<(), String> {
        self.signature.validate()?;
        checked_output_element_count(&self.signature)?;
        if self.matrix_ptr == 0 || self.activation_ptr == 0 || self.output_ptr == 0 {
            return Err("IQ1_S launch contains a null CUDA pointer".into());
        }
        self.signature.matrix_storage_bytes()?;
        self.signature.activation_storage_bytes()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CapturedLaunch {
    pub(crate) launch: LogicalLaunch,
    pub(crate) matrix: Arc<MatrixSource>,
    packed_activations: Arc<[u8]>,
}

#[derive(Debug, Clone)]
pub(crate) struct CapturedActivationLaunch {
    pub(crate) launch: LogicalLaunch,
    packed_activations: Arc<[u8]>,
}

impl CapturedActivationLaunch {
    pub(crate) fn packed_activations(&self) -> &[u8] {
        &self.packed_activations
    }
}

pub(crate) trait VecCaptureCopies {
    fn copy_matrix(&self, pointer: usize, bytes: usize) -> Result<Vec<u8>, String>;
    fn copy_activation(&self, pointer: usize, bytes: usize) -> Result<Vec<u8>, String>;
}

struct CudaVecCaptureCopies;

impl VecCaptureCopies for CudaVecCaptureCopies {
    fn copy_matrix(&self, pointer: usize, bytes: usize) -> Result<Vec<u8>, String> {
        unsafe { copy_cuda_to_host(pointer, bytes) }.map_err(|error| error.to_string())
    }

    fn copy_activation(&self, pointer: usize, bytes: usize) -> Result<Vec<u8>, String> {
        unsafe { copy_cuda_to_host(pointer, bytes) }.map_err(|error| error.to_string())
    }
}

impl CapturedLaunch {
    pub(crate) fn activation_blocks(
        &self,
    ) -> Result<impl Iterator<Item = Result<Q8_1MmqBlock, String>> + '_, String> {
        iter_q8_1_mmq(&self.launch.signature, &self.packed_activations)
    }

    pub(crate) fn q8_group(
        &self,
        batch_index: usize,
        global_group: usize,
    ) -> Result<Q8_1Block, String> {
        let layout = self.launch.signature.q8_mmq_layout()?;
        if batch_index >= layout.active_batch_count {
            return Err("Q8 batch index is outside the captured activation".into());
        }
        let block_index = global_group / 4;
        let subblock = global_group % 4;
        if block_index >= layout.q8_record_count_per_k {
            return Err("Q8 group index is outside the captured activation K records".into());
        }
        let range = layout.record_range(block_index, batch_index)?;
        let packed = self
            .packed_activations
            .get(range)
            .ok_or("Q8 group is outside the captured activation")?;
        Ok(Q8_1MmqBlock::parse(packed)?.subblocks[subblock])
    }
}

static COMPONENT_CACHE: OnceLock<Mutex<ComponentCache>> = OnceLock::new();

pub(crate) fn capture_from_host(
    launch: LogicalLaunch,
    packed_matrix: &[u8],
    packed_activations: &[u8],
    grid: &GridTable,
) -> Result<CapturedLaunch, String> {
    launch.validate_before_copy()?;
    let matrix_bytes = launch.signature.matrix_storage_bytes()?;
    let activation_bytes = launch.signature.activation_storage_bytes()?;
    if packed_matrix.len() != matrix_bytes || packed_activations.len() != activation_bytes {
        return Err("captured CUDA storage length does not match the qualified signature".into());
    }
    // Force a complete finite DS4 validation while retaining the packed bytes;
    // execution can subsequently iterate the groups without a dense activation.
    for block in iter_q8_1_mmq(&launch.signature, packed_activations)? {
        block?;
    }
    let identity = MatrixCacheIdentity {
        matrix_ptr: launch.matrix_ptr,
        signature: launch.signature.clone(),
        allocation_generation: launch.allocation_generation,
        content_hash: launch.content_hash,
    };
    let cache = COMPONENT_CACHE.get_or_init(|| Mutex::new(ComponentCache::default()));
    let cached = {
        let guard = cache.lock().map_err(|_| "component cache poisoned")?;
        guard.get(&identity)
    };
    let matrix = if let Some(cached) = cached {
        cached
    } else {
        let built = MatrixSource::new(
            launch.signature.clone(),
            Arc::from(packed_matrix),
            Arc::new(*grid),
        )?;
        let mut guard = cache.lock().map_err(|_| "component cache poisoned")?;
        if let Some(cached) = guard.get(&identity) {
            cached
        } else {
            guard.insert(identity, built.clone());
            built
        }
    };
    Ok(CapturedLaunch {
        launch,
        matrix,
        packed_activations: Arc::from(packed_activations),
    })
}

pub(crate) fn capture_activation_from_host(
    launch: LogicalLaunch,
    packed_activations: &[u8],
) -> Result<CapturedActivationLaunch, String> {
    launch.validate_before_copy()?;
    if launch.content_hash == [0; 32] {
        return Err("persistent IQ1_S capture requires a registered nonzero content hash".into());
    }
    let activation_bytes = launch.signature.activation_storage_bytes()?;
    if packed_activations.len() != activation_bytes {
        return Err(
            "captured CUDA activation length does not match the qualified signature".into(),
        );
    }
    for block in iter_q8_1_mmq(&launch.signature, packed_activations)? {
        block?;
    }
    Ok(CapturedActivationLaunch {
        launch,
        packed_activations: Arc::from(packed_activations),
    })
}

pub(crate) unsafe fn capture_launch(launch: LogicalLaunch) -> Result<CapturedLaunch, String> {
    let mut launch = launch;
    launch.validate_before_copy()?;
    let matrix = copy_cuda_to_host(launch.matrix_ptr, launch.signature.matrix_storage_bytes()?)
        .map_err(|error| error.to_string())?;
    let activations = copy_cuda_to_host(
        launch.activation_ptr,
        launch.signature.activation_storage_bytes()?,
    )
    .map_err(|error| error.to_string())?;
    if launch.content_hash == [0; 32] {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        matrix.len().hash(&mut hasher);
        matrix.hash(&mut hasher);
        launch.content_hash[..8].copy_from_slice(&hasher.finish().to_le_bytes());
    }
    let grid = validated_grid(None)?;
    capture_from_host(launch, &matrix, &activations, &grid)
}

pub(crate) fn canonicalize_q8_1_vector(
    signature: &GgmlType19VecSignature,
    packed: &[u8],
) -> Result<Vec<u8>, String> {
    signature.validate()?;
    let expected = signature.activation_storage_bytes()?;
    if packed.len() != expected {
        return Err(format!(
            "Q8_1 vector storage must be exactly {expected} bytes"
        ));
    }
    let groups = packed.len() / (4 * Q8_1_BLOCK_BYTES);
    let mut canonical = vec![0_u8; groups * Q8_1_MMQ_BYTES];
    for group_index in 0..groups {
        let source =
            &packed[group_index * 4 * Q8_1_BLOCK_BYTES..(group_index + 1) * 4 * Q8_1_BLOCK_BYTES];
        let destination =
            &mut canonical[group_index * Q8_1_MMQ_BYTES..(group_index + 1) * Q8_1_MMQ_BYTES];
        for block_index in 0..4 {
            let source_block =
                &source[block_index * Q8_1_BLOCK_BYTES..(block_index + 1) * Q8_1_BLOCK_BYTES];
            let scale_offset = block_index * 4;
            destination[scale_offset..scale_offset + 4].copy_from_slice(&source_block[..4]);
            let values_offset = 16 + block_index * 32;
            destination[values_offset..values_offset + 32].copy_from_slice(&source_block[4..]);
        }
    }
    Ok(canonical)
}

pub(crate) fn capture_vec_activation_with(
    matrix_ptr: usize,
    activation_ptr: usize,
    output_ptr: usize,
    allocation_generation: u64,
    content_hash: [u8; 32],
    signature: GgmlType19VecSignature,
    copies: &impl VecCaptureCopies,
) -> Result<CapturedActivationLaunch, String> {
    let signature = signature.validate()?;
    if matrix_ptr == 0 || activation_ptr == 0 || output_ptr == 0 {
        return Err("IQ1_S vector launch contains a null CUDA pointer".into());
    }
    if content_hash == [0; 32] {
        return Err("persistent IQ1_S capture requires a registered nonzero content hash".into());
    }
    let raw_activations =
        copies.copy_activation(activation_ptr, signature.activation_storage_bytes()?)?;
    let activations = canonicalize_q8_1_vector(&signature, &raw_activations)?;
    capture_activation_from_host(
        LogicalLaunch {
            matrix_ptr,
            activation_ptr,
            output_ptr,
            allocation_generation,
            content_hash,
            signature: signature.mmq_signature()?,
        },
        &activations,
    )
}

pub(crate) unsafe fn capture_vec_activation(
    matrix_ptr: usize,
    activation_ptr: usize,
    output_ptr: usize,
    allocation_generation: u64,
    content_hash: [u8; 32],
    signature: GgmlType19VecSignature,
) -> Result<CapturedActivationLaunch, String> {
    capture_vec_activation_with(
        matrix_ptr,
        activation_ptr,
        output_ptr,
        allocation_generation,
        content_hash,
        signature,
        &CudaVecCaptureCopies,
    )
}

pub(crate) unsafe fn capture_vec_launch(
    matrix_ptr: usize,
    activation_ptr: usize,
    output_ptr: usize,
    allocation_generation: u64,
    content_hash: [u8; 32],
    signature: GgmlType19VecSignature,
) -> Result<CapturedLaunch, String> {
    let signature = signature.validate()?;
    if matrix_ptr == 0 || activation_ptr == 0 || output_ptr == 0 {
        return Err("IQ1_S vector launch contains a null CUDA pointer".into());
    }
    let matrix = copy_cuda_to_host(matrix_ptr, signature.matrix_storage_bytes()?)
        .map_err(|error| error.to_string())?;
    let raw_activations = copy_cuda_to_host(activation_ptr, signature.activation_storage_bytes()?)
        .map_err(|error| error.to_string())?;
    let activations = canonicalize_q8_1_vector(&signature, &raw_activations)?;
    let mmq_signature = signature.mmq_signature()?;
    let mut launch = LogicalLaunch {
        matrix_ptr,
        activation_ptr,
        output_ptr,
        allocation_generation,
        content_hash,
        signature: mmq_signature,
    };
    if launch.content_hash == [0; 32] {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        matrix.len().hash(&mut hasher);
        matrix.hash(&mut hasher);
        launch.content_hash[..8].copy_from_slice(&hasher.finish().to_le_bytes());
    }
    let grid = validated_grid(None)?;
    capture_from_host(launch, &matrix, &activations, &grid)
}

fn output_bytes(captured: &CapturedLaunch, outputs: &[f32]) -> Result<Vec<u8>, String> {
    let expected = checked_output_element_count(&captured.launch.signature)?;
    if outputs.len() != expected || outputs.iter().any(|value| !value.is_finite()) {
        return Err("output must contain the full finite qualified f32 result".into());
    }
    let byte_count = outputs
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("output size overflow")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_count)
        .map_err(|error| format!("unable to reserve {byte_count} output bytes: {error}"))?;
    for value in outputs {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) unsafe fn copy_outputs_to_cuda(
    captured: &CapturedLaunch,
    outputs: &[f32],
) -> Result<(), String> {
    let bytes = output_bytes(captured, outputs)?;
    copy_host_to_cuda(captured.launch.output_ptr, &bytes).map_err(|error| error.to_string())
}

pub(crate) trait DaxAccess {
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<usize, String>;
    fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<usize, String>;
}

fn validate_v3_memory_setting(value: Option<&str>) -> Result<(), String> {
    match value.map(str::trim) {
        None | Some("") | Some("dax") => Ok(()),
        Some(other) => Err(format!(
            "unsupported HETGPU_CXL_TMATMUL_V3_MEMORY={other:?}; only dax is allowed"
        )),
    }
}

pub(crate) struct FileDaxAccess {
    mapping: NonNull<u8>,
    length: usize,
    io_lock: Mutex<()>,
}

impl FileDaxAccess {
    pub(crate) fn map(bound: &BoundDaxFile) -> Result<Self, String> {
        let length = usize::try_from(bound.length())
            .map_err(|_| format!("DAX mapping length does not fit usize: {}", bound.length()))?;
        if length == 0 || length > isize::MAX as usize {
            return Err(format!("invalid DAX mapping length: {length}"));
        }
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                bound.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(format!(
                "mmap bound DAX generation={} length=0x{length:x}: {}",
                bound.generation(),
                std::io::Error::last_os_error()
            ));
        }
        let mapping = NonNull::new(mapped.cast::<u8>())
            .ok_or_else(|| "mmap bound DAX returned null".to_string())?;
        Ok(Self {
            mapping,
            length,
            io_lock: Mutex::new(()),
        })
    }

    fn checked_range(&self, offset: u64, length: usize) -> Result<usize, String> {
        let offset = usize::try_from(offset).map_err(|_| "DAX offset does not fit usize")?;
        let end = offset.checked_add(length).ok_or("DAX range overflow")?;
        if end > self.length {
            return Err(format!(
                "DAX range 0x{offset:x}..0x{end:x} exceeds mapping length 0x{:x}",
                self.length
            ));
        }
        Ok(offset)
    }
}

impl DaxAccess for FileDaxAccess {
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<usize, String> {
        let offset = self.checked_range(offset, bytes.len())?;
        let _guard = self
            .io_lock
            .lock()
            .map_err(|_| "DAX mapping lock poisoned")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapping.as_ptr().add(offset),
                bytes.len(),
            );
        }
        Ok(bytes.len())
    }
    fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<usize, String> {
        let offset = self.checked_range(offset, bytes.len())?;
        let _guard = self
            .io_lock
            .lock()
            .map_err(|_| "DAX mapping lock poisoned")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mapping.as_ptr().add(offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(bytes.len())
    }
}

impl Drop for FileDaxAccess {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping.as_ptr().cast(), self.length);
        }
    }
}

pub(crate) trait OutputCopier {
    unsafe fn copy(&self, pointer: usize, bytes: &[u8]) -> Result<(), String>;
}

pub(crate) struct CudaOutputCopier;
impl OutputCopier for CudaOutputCopier {
    unsafe fn copy(&self, pointer: usize, bytes: &[u8]) -> Result<(), String> {
        copy_host_to_cuda(pointer, bytes).map_err(|error| error.to_string())
    }
}

#[derive(Debug)]
pub(crate) struct ExecutionResult {
    pub(crate) outputs: Vec<f32>,
    pub(crate) physical: Vec<PhysicalResult>,
    pub(crate) raw_components: Vec<Vec<i64>>,
    pub(crate) scheduler: SchedulerReport,
}

#[derive(Debug, Clone, Copy)]
struct PlannedComponent {
    kind: ComponentKind,
    geometry: TileGeometry,
    group32: usize,
    matrix_offset: u64,
    input_offset: u64,
    output_offset: u64,
}

fn dax_write_exact(dax: &dyn DaxAccess, offset: u64, bytes: &[u8]) -> Result<(), String> {
    let written = dax.write(offset, bytes)?;
    if written != bytes.len() {
        return Err(format!(
            "short DAX write at {offset}: {written}/{}",
            bytes.len()
        ));
    }
    Ok(())
}

fn dax_read_exact(dax: &dyn DaxAccess, offset: u64, bytes: &mut [u8]) -> Result<(), String> {
    let read = dax.read(offset, bytes)?;
    if read != bytes.len() {
        return Err(format!(
            "short DAX read at {offset}: {read}/{}",
            bytes.len()
        ));
    }
    Ok(())
}

fn aligned(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("invalid DAX alignment".into());
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or("DAX layout overflow".into())
}

fn cleanup_v3_leases<I: IoctlOps>(
    session: &mut V3Session<I>,
    leases: &[(&str, BufferLease)],
) -> Vec<String> {
    let mut errors = Vec::new();
    for &(name, lease) in leases {
        if let Err(error) = session.unregister_buffer(lease) {
            errors.push(format!("{name}: {error}"));
        }
    }
    errors
}

fn cleanup_matrix_leases<I: IoctlOps>(
    session: &mut V3Session<I>,
    matrices: &[BufferLease],
) -> Vec<String> {
    let mut errors = Vec::new();
    for &matrix in matrices.iter().rev() {
        if let Err(error) = session.unregister_buffer(matrix) {
            errors.push(format!("matrix handle {}: {error}", matrix.handle()));
        }
    }
    errors
}

fn append_cleanup_error(error: String, cleanup_errors: Vec<String>) -> String {
    if cleanup_errors.is_empty() {
        error
    } else {
        format!(
            "{error}; executor buffer cleanup also failed: {}",
            cleanup_errors.join("; ")
        )
    }
}

pub(crate) fn execute_captured_with<I: IoctlOps>(
    captured: &CapturedLaunch,
    session: &mut V3Session<I>,
    dax: &dyn DaxAccess,
    output_copier: &dyn OutputCopier,
    base_dpa: u64,
) -> Result<ExecutionResult, String> {
    captured.launch.validate_before_copy()?;
    let caps = session.caps();
    let alignment = u64::from(caps.dax_alignment_bytes);
    if !base_dpa.is_multiple_of(alignment) {
        return Err("executor base DPA is not aligned".into());
    }
    let logical_batch = u32::try_from(captured.launch.signature.ne11)
        .map_err(|_| "logical batch does not fit u32")?;
    let logical_batch_usize =
        usize::try_from(logical_batch).map_err(|_| "logical batch does not fit usize")?;
    let batch_config = BatchSchedulerConfig::from_env(caps.max_batch)?;
    let batch_plan = batch_config.plan(logical_batch)?;
    let task_count = captured.matrix.component_iter()?.len();
    if task_count == 0 {
        return Err("executor has no physical components".into());
    }
    let task_count_u64 = u64::try_from(task_count).map_err(|_| "component count overflow")?;
    let descriptor_count =
        checked_execution_descriptor_count(task_count, batch_plan.slices().len())?;
    let matrix_slot = aligned(
        u64::try_from(TILE_PACKED_BYTES).map_err(|_| "matrix slot size overflow")?,
        alignment,
    )?;
    let input_row_len = TILE_DIM.checked_mul(2).ok_or("input row size overflow")?;
    let output_row_len = TILE_DIM.checked_mul(8).ok_or("output row size overflow")?;
    let input_row_bytes =
        u64::try_from(input_row_len).map_err(|_| "input row size does not fit u64")?;
    let output_row_bytes =
        u64::try_from(output_row_len).map_err(|_| "output row size does not fit u64")?;
    let input_row_stride = aligned(input_row_bytes, alignment)?;
    let output_row_stride = aligned(output_row_bytes, alignment)?;
    let input_component_bytes = input_row_stride
        .checked_mul(u64::from(logical_batch))
        .ok_or("component input slab overflow")?;
    let output_component_bytes = output_row_stride
        .checked_mul(u64::from(logical_batch))
        .ok_or("component output slab overflow")?;
    let matrix_bytes = matrix_slot
        .checked_mul(task_count_u64)
        .ok_or("matrix region overflow")?;
    let input_base = aligned(
        base_dpa
            .checked_add(matrix_bytes)
            .ok_or("input base overflow")?,
        alignment,
    )?;
    let input_bytes = input_component_bytes
        .checked_mul(task_count_u64)
        .ok_or("input region overflow")?;
    let output_base = aligned(
        input_base
            .checked_add(input_bytes)
            .ok_or("output base overflow")?,
        alignment,
    )?;
    let output_region_bytes = output_component_bytes
        .checked_mul(task_count_u64)
        .ok_or("output region overflow")?;
    let end = output_base
        .checked_add(output_region_bytes)
        .ok_or("executor DAX range overflow")?;
    if end > caps.dax_bytes {
        return Err("executor DAX layout exceeds live capacity".into());
    }

    let zero_output_row = vec![0_u8; output_row_len];
    let mut planned = Vec::new();
    planned
        .try_reserve_exact(task_count)
        .map_err(|error| format!("unable to reserve {task_count} planned components: {error}"))?;
    for (index, component) in captured.matrix.component_iter()?.enumerate() {
        let component = component?;
        let index_u64 = u64::try_from(index).map_err(|_| "component index overflow")?;
        let matrix_offset = matrix_slot
            .checked_mul(index_u64)
            .ok_or("matrix offset overflow")?;
        let input_offset = input_component_bytes
            .checked_mul(index_u64)
            .ok_or("input offset overflow")?;
        let output_offset = output_component_bytes
            .checked_mul(index_u64)
            .ok_or("output offset overflow")?;
        let matrix_dpa = base_dpa
            .checked_add(matrix_offset)
            .ok_or("matrix DPA overflow")?;
        dax_write_exact(dax, matrix_dpa, &component.pack_ternary2()?)?;
        let global_group =
            component.geometry.column_tile * (TILE_DIM / GROUP_VALUES) + component.group32;
        for batch_index in 0..logical_batch_usize {
            let batch_u64 = u64::try_from(batch_index).map_err(|_| "batch index overflow")?;
            let input_row_offset = input_offset
                .checked_add(
                    input_row_stride
                        .checked_mul(batch_u64)
                        .ok_or("input row offset overflow")?,
                )
                .ok_or("input row offset overflow")?;
            let output_row_offset = output_offset
                .checked_add(
                    output_row_stride
                        .checked_mul(batch_u64)
                        .ok_or("output row offset overflow")?,
                )
                .ok_or("output row offset overflow")?;
            let q8 = captured.q8_group(batch_index, global_group)?;
            let mut input_row = vec![0_u8; input_row_len];
            let encoded = encode_q8_8(&q8.qs);
            let local_column = component
                .group32
                .checked_mul(GROUP_VALUES)
                .ok_or("local input column overflow")?;
            for (column, value) in encoded.into_iter().enumerate() {
                let offset = local_column
                    .checked_add(column)
                    .and_then(|element| element.checked_mul(2))
                    .ok_or("input element offset overflow")?;
                let end = offset.checked_add(2).ok_or("input element end overflow")?;
                let destination = input_row
                    .get_mut(offset..end)
                    .ok_or("input element exceeds staged row")?;
                destination.copy_from_slice(&value.to_le_bytes());
            }
            dax_write_exact(
                dax,
                input_base
                    .checked_add(input_row_offset)
                    .ok_or("input DPA overflow")?,
                &input_row,
            )?;
            dax_write_exact(
                dax,
                output_base
                    .checked_add(output_row_offset)
                    .ok_or("output DPA overflow")?,
                &zero_output_row,
            )?;
        }
        planned.push(PlannedComponent {
            kind: component.kind,
            geometry: component.geometry,
            group32: component.group32,
            matrix_offset,
            input_offset,
            output_offset,
        });
    }

    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(task_count)
        .map_err(|error| format!("unable to reserve {task_count} matrix leases: {error}"))?;
    for item in &planned {
        let matrix_dpa = base_dpa
            .checked_add(item.matrix_offset)
            .ok_or("matrix registration DPA overflow")?;
        let registered = match session.register_buffer(
            matrix_dpa,
            matrix_slot,
            BUFFER_TERNARY2,
            BUFFER_READ | BUFFER_MATRIX,
        ) {
            Ok(matrix) => matrix,
            Err(error) => {
                let cleanup = cleanup_matrix_leases(session, &matrices);
                return Err(append_cleanup_error(error, cleanup));
            }
        };
        let committed = match session.commit_buffer(registered) {
            Ok(matrix) => matrix,
            Err(error) => {
                let mut cleanup = cleanup_v3_leases(session, &[("matrix", registered)]);
                cleanup.extend(cleanup_matrix_leases(session, &matrices));
                return Err(append_cleanup_error(error, cleanup));
            }
        };
        matrices.push(committed);
    }
    let input = match session.register_buffer(input_base, input_bytes, BUFFER_Q8_8_S16, BUFFER_READ)
    {
        Ok(input) => input,
        Err(error) => {
            let cleanup = cleanup_matrix_leases(session, &matrices);
            return Err(append_cleanup_error(error, cleanup));
        }
    };
    let output = match session.register_buffer(
        output_base,
        output_region_bytes,
        BUFFER_RAW_S64,
        BUFFER_WRITE,
    ) {
        Ok(output) => output,
        Err(error) => {
            let mut cleanup = cleanup_v3_leases(session, &[("input", input)]);
            cleanup.extend(cleanup_matrix_leases(session, &matrices));
            return Err(append_cleanup_error(error, cleanup));
        }
    };
    let execution = (|| -> Result<ExecutionResult, String> {
        let mut tasks = Vec::new();
        tasks.try_reserve_exact(descriptor_count).map_err(|error| {
            format!("unable to reserve {descriptor_count} execution descriptors: {error}")
        })?;
        for (component_index, item) in planned.iter().enumerate() {
            let leases = TaskLeases {
                matrix: *matrices
                    .get(component_index)
                    .ok_or("missing committed component matrix")?,
                input,
                output,
            };
            for slice in batch_plan.slices() {
                let request_id =
                    u64::try_from(tasks.len().checked_add(1).ok_or("request ID overflow")?)
                        .map_err(|_| "request ID overflow")?;
                let slice_first_u64 = u64::from(slice.first());
                let input_offset = item
                    .input_offset
                    .checked_add(
                        input_row_stride
                            .checked_mul(slice_first_u64)
                            .ok_or("task input offset overflow")?,
                    )
                    .ok_or("task input offset overflow")?;
                let output_offset = item
                    .output_offset
                    .checked_add(
                        output_row_stride
                            .checked_mul(slice_first_u64)
                            .ok_or("task output offset overflow")?,
                    )
                    .ok_or("task output offset overflow")?;
                let mut task = build_task(
                    request_id,
                    item.geometry,
                    leases,
                    slice.count(),
                    output_offset,
                    alignment,
                )?;
                task.input_offset = input_offset;
                tasks.push(PlannedTask {
                    kind: item.kind,
                    geometry: item.geometry,
                    group32: item.group32,
                    batch_first: slice.first(),
                    batch_count: slice.count(),
                    leases,
                    task,
                });
            }
        }
        let (physical, scheduler) = run_component_tasks(session, &tasks)?;

        let raw_count = task_count
            .checked_mul(logical_batch_usize)
            .ok_or("raw component count overflow")?;
        let mut raw_outputs = Vec::new();
        raw_outputs.try_reserve_exact(raw_count).map_err(|error| {
            format!("unable to reserve {raw_count} raw component rows: {error}")
        })?;
        for item in &planned {
            for batch_index in 0..logical_batch_usize {
                let batch_u64 = u64::try_from(batch_index).map_err(|_| "batch index overflow")?;
                let row_offset = item
                    .output_offset
                    .checked_add(
                        output_row_stride
                            .checked_mul(batch_u64)
                            .ok_or("raw output row offset overflow")?,
                    )
                    .ok_or("raw output row offset overflow")?;
                let mut bytes = vec![0_u8; output_row_len];
                dax_read_exact(
                    dax,
                    output_base
                        .checked_add(row_offset)
                        .ok_or("raw output DPA overflow")?,
                    &mut bytes,
                )?;
                let valid_bytes = item
                    .geometry
                    .valid_out
                    .checked_mul(8)
                    .ok_or("raw output byte count overflow")?;
                let mut raw = Vec::with_capacity(item.geometry.valid_out);
                for chunk in bytes
                    .get(..valid_bytes)
                    .ok_or("raw output exceeds staged row")?
                    .chunks_exact(8)
                {
                    raw.push(i64::from_le_bytes(
                        chunk.try_into().map_err(|_| "invalid raw output width")?,
                    ));
                }
                raw_outputs.push(raw);
            }
        }
        let output_rows = usize::try_from(captured.launch.signature.ne0)
            .map_err(|_| "output row count overflow")?;
        let output_count = logical_batch_usize
            .checked_mul(output_rows)
            .ok_or("output count overflow")?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(output_count)
            .map_err(|error| format!("unable to reserve {output_count} output values: {error}"))?;
        outputs.resize(output_count, 0_f32);
        for pair_index in (0..planned.len()).step_by(2) {
            let grid_plan = planned[pair_index];
            let delta_plan = *planned
                .get(pair_index + 1)
                .ok_or("missing delta component")?;
            if grid_plan.kind != ComponentKind::Grid
                || delta_plan.kind != ComponentKind::Delta
                || grid_plan.geometry != delta_plan.geometry
                || grid_plan.group32 != delta_plan.group32
            {
                return Err("grid/delta physical component order is inconsistent".into());
            }
            let global_group =
                grid_plan.geometry.column_tile * (TILE_DIM / GROUP_VALUES) + grid_plan.group32;
            for batch_index in 0..logical_batch_usize {
                let q8 = captured.q8_group(batch_index, global_group)?;
                let grid_raw_index = pair_index
                    .checked_mul(logical_batch_usize)
                    .and_then(|base| base.checked_add(batch_index))
                    .ok_or("grid raw component index overflow")?;
                let delta_raw_index = (pair_index + 1)
                    .checked_mul(logical_batch_usize)
                    .and_then(|base| base.checked_add(batch_index))
                    .ok_or("delta raw component index overflow")?;
                for row_in_tile in 0..grid_plan.geometry.valid_out {
                    let global_row = grid_plan
                        .geometry
                        .row_tile
                        .checked_mul(TILE_DIM)
                        .and_then(|base| base.checked_add(row_in_tile))
                        .ok_or("global output row overflow")?;
                    let output_index = batch_index
                        .checked_mul(output_rows)
                        .and_then(|base| base.checked_add(global_row))
                        .ok_or("batch-major output index overflow")?;
                    let (d, group) = captured.matrix.group(global_row, global_group)?;
                    let grid_raw = *raw_outputs
                        .get(grid_raw_index)
                        .and_then(|raw| raw.get(row_in_tile))
                        .ok_or("missing grid raw output")?;
                    let delta_raw = *raw_outputs
                        .get(delta_raw_index)
                        .and_then(|raw| raw.get(row_in_tile))
                        .ok_or("missing delta raw output")?;
                    let contribution = reconstruct_from_raw(&group, d, &q8, grid_raw, delta_raw)?;
                    let output = outputs
                        .get_mut(output_index)
                        .ok_or("batch-major output exceeds allocation")?;
                    *output = (*output + contribution) as f32;
                    if !output.is_finite() {
                        return Err("accumulated output is nonfinite".into());
                    }
                }
            }
        }
        let output_bytes_host = output_bytes(captured, &outputs)?;
        unsafe {
            output_copier.copy(captured.launch.output_ptr, &output_bytes_host)?;
        }
        Ok(ExecutionResult {
            outputs,
            physical,
            raw_components: raw_outputs,
            scheduler,
        })
    })();

    let mut cleanup_errors = cleanup_v3_leases(session, &[("output", output), ("input", input)]);
    cleanup_errors.extend(cleanup_matrix_leases(session, &matrices));
    match (execution, cleanup_errors.is_empty()) {
        (Ok(result), true) => Ok(result),
        (Ok(_), false) => Err(format!(
            "executor buffer cleanup failed: {}",
            cleanup_errors.join("; ")
        )),
        (Err(error), true) => Err(error),
        (Err(error), false) => Err(format!(
            "{error}; executor buffer cleanup also failed: {}",
            cleanup_errors.join("; ")
        )),
    }
}

pub(crate) unsafe fn execute_captured(
    captured: &CapturedLaunch,
    control_path: &Path,
    dax_path: &Path,
    base_dpa: u64,
) -> Result<ExecutionResult, String> {
    validate_v3_memory_setting(
        std::env::var("HETGPU_CXL_TMATMUL_V3_MEMORY")
            .ok()
            .as_deref(),
    )?;
    let (mut session, bound_dax) = V3Session::open(control_path, dax_path)?;
    let dax = FileDaxAccess::map(&bound_dax)?;
    execute_captured_with(captured, &mut session, &dax, &CudaOutputCopier, base_dpa)
}

#[derive(Serialize)]
struct ExecutionFixture {
    libggml_path: String,
    iq1s_hex: String,
    q8_1_mmq_hex: String,
    raw_components: Vec<Vec<i64>>,
    submission_ids: Vec<u64>,
    request_ids: Vec<u64>,
    outputs_f32_bits: Vec<u32>,
}

pub(crate) fn execution_fixture_json(
    captured: &CapturedLaunch,
    result: &ExecutionResult,
    libggml_path: &Path,
) -> Result<String, String> {
    if result.outputs.iter().any(|value| !value.is_finite()) {
        return Err("execution fixture contains nonfinite output".into());
    }
    let resolved_oracle = libggml_path
        .canonicalize()
        .map_err(|error| format!("canonical fixture libggml path: {error}"))?;
    serde_json::to_string(&ExecutionFixture {
        libggml_path: resolved_oracle.display().to_string(),
        iq1s_hex: hex(&captured.matrix.packed),
        q8_1_mmq_hex: hex(&captured.packed_activations),
        raw_components: result.raw_components.clone(),
        submission_ids: result
            .physical
            .iter()
            .map(|item| item.completed.submission_id())
            .collect(),
        request_ids: result
            .physical
            .iter()
            .map(|item| item.completed.task().request_id)
            .collect(),
        outputs_f32_bits: result.outputs.iter().map(|value| value.to_bits()).collect(),
    })
    .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskLeases {
    pub(crate) matrix: BufferLease,
    pub(crate) input: BufferLease,
    pub(crate) output: BufferLease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalResult {
    pub(crate) kind: ComponentKind,
    pub(crate) row_tile: usize,
    pub(crate) column_tile: usize,
    pub(crate) group32: usize,
    pub(crate) batch_first: u32,
    pub(crate) batch_count: u32,
    pub(crate) completed: CompletedTaskV3,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlannedTask {
    kind: ComponentKind,
    geometry: TileGeometry,
    group32: usize,
    batch_first: u32,
    batch_count: u32,
    leases: TaskLeases,
    task: TaskV3,
}

pub(crate) fn run_component_tasks<I: IoctlOps>(
    session: &mut V3Session<I>,
    components: &[PlannedTask],
) -> Result<(Vec<PhysicalResult>, SchedulerReport), String> {
    let mut tasks = Vec::new();
    tasks.try_reserve_exact(components.len()).map_err(|error| {
        format!(
            "unable to reserve {} submitted descriptors: {error}",
            components.len()
        )
    })?;
    tasks.extend(components.iter().map(|item| item.task));
    for (task, component) in tasks.iter().zip(components) {
        if task.matrix_handle != component.leases.matrix.handle()
            || task.matrix_generation != component.leases.matrix.generation()
            || task.input_handle != component.leases.input.handle()
            || task.output_handle != component.leases.output.handle()
        {
            return Err("planned V3 task does not match committed leases".into());
        }
    }
    let completed = session.run_tasks(&tasks)?;
    if completed.len() != components.len() {
        return Err("V3 returned an incomplete component set".into());
    }
    let scheduler = SchedulerReport::from_completions(&completed, MAX_LANES)?;
    let mut physical = Vec::new();
    physical
        .try_reserve_exact(components.len())
        .map_err(|error| {
            format!(
                "unable to reserve {} physical results: {error}",
                components.len()
            )
        })?;
    physical.extend(
        components
            .iter()
            .zip(completed)
            .map(|(component, completed)| PhysicalResult {
                kind: component.kind,
                row_tile: component.geometry.row_tile,
                column_tile: component.geometry.column_tile,
                group32: component.group32,
                batch_first: component.batch_first,
                batch_count: component.batch_count,
                completed,
            }),
    );
    Ok((physical, scheduler))
}

pub(crate) fn build_task(
    request_id: u64,
    geometry: TileGeometry,
    leases: TaskLeases,
    batch: u32,
    output_offset: u64,
    dax_alignment: u64,
) -> Result<TaskV3, String> {
    if dax_alignment == 0 || !dax_alignment.is_power_of_two() || !output_offset.is_multiple_of(8) {
        return Err("invalid DAX/output alignment".into());
    }
    let stride = align_up((TILE_DIM * 2) as u64, dax_alignment)?;
    let mut task = TaskV3::default();
    task.request_id = request_id;
    task.lane = LANE_ANY;
    task.batch = batch;
    task.valid_out = geometry.valid_out as u32;
    task.valid_in = geometry.valid_in as u32;
    task.matrix_handle = leases.matrix.handle();
    task.input_handle = leases.input.handle();
    task.output_handle = leases.output.handle();
    task.matrix_generation = leases.matrix.generation();
    task.input_stride_bytes = stride;
    task.output_offset = output_offset;
    task.output_stride_bytes = align_up((TILE_DIM * 8) as u64, dax_alignment)?;
    Ok(task)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("invalid alignment".into());
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or("alignment overflow".into())
}

#[derive(Serialize)]
struct Fixture<'a> {
    iq1s_hex: String,
    q8_1_mmq_hex: String,
    raw_components: &'a [(i64, i64)],
    outputs_f32_bits: Vec<u32>,
}

pub(crate) fn fixture_json(
    iq1s: &[u8],
    q8: &[u8],
    raw: &[(i64, i64)],
    output: &[f32],
) -> Result<String, String> {
    if output.iter().any(|value| !value.is_finite()) {
        return Err("fixture output is nonfinite".into());
    }
    serde_json::to_string(&Fixture {
        iq1s_hex: hex(iq1s),
        q8_1_mmq_hex: hex(q8),
        raw_components: raw,
        outputs_f32_bits: output.iter().map(|value| value.to_bits()).collect(),
    })
    .map_err(|error| error.to_string())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = fraction.leading_zeros() - 22;
            let normalized = (fraction << (leading + 1)) & 0x03ff;
            sign | ((127 - 15 - leading) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(f32_bits)
}

fn f32_to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x80_0000;
        let shift = 14 - half_exp;
        let mut rounded = mant >> shift;
        let remainder = mant & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
    }
    let mut exp = half_exp as u16;
    if rounded == 0x400 {
        rounded = 0;
        exp += 1;
    }
    if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | (exp << 10) | rounded as u16
    }
}

fn round_to_half(value: f32) -> Result<f32, String> {
    let rounded = half_to_f32(f32_to_half_bits(value));
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err("value does not round to finite binary16".into())
    }
}

#[repr(C)]
struct GgmlInitParams {
    mem_size: usize,
    mem_buffer: *mut libc::c_void,
    no_alloc: bool,
}
type DequantizeIq1s = unsafe extern "C" fn(*const libc::c_void, *mut libc::c_void, i64);
type GgmlInit = unsafe extern "C" fn(GgmlInitParams) -> *mut libc::c_void;
type GgmlFree = unsafe extern "C" fn(*mut libc::c_void);

struct OracleLibrary {
    handle: *mut libc::c_void,
    dequantize: DequantizeIq1s,
}
unsafe impl Send for OracleLibrary {}
unsafe impl Sync for OracleLibrary {}
impl Drop for OracleLibrary {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

static GRID_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<GridTable>>>> = OnceLock::new();
static GRID_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();

fn select_libggml_path(path: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = path {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("HETGPU_LIBGGML") {
        return Ok(PathBuf::from(path));
    }
    if std::env::var("HETGPU_QWEN_IQ1S_STRICT").as_deref() == Ok("1") {
        return Err("strict Qwen IQ1_S mode requires verified HETGPU_LIBGGML".to_string());
    }
    Ok(PathBuf::from(DEFAULT_LIBGGML))
}

pub(crate) fn validated_grid(path: Option<&Path>) -> Result<Arc<GridTable>, String> {
    let selected = select_libggml_path(path)?;
    let path = selected
        .canonicalize()
        .map_err(|error| format!("libggml {}: {error}", selected.display()))?;
    let cache = GRID_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(value) = cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .get(&path)
        .cloned()
    {
        return Ok(value);
    }
    let key_lock = {
        let locks = GRID_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
        locks
            .lock()
            .map_err(|_| "grid lock map poisoned")?
            .entry(path.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _guard = key_lock.lock().map_err(|_| "grid key lock poisoned")?;
    if let Some(value) = cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .get(&path)
        .cloned()
    {
        return Ok(value);
    }
    let oracle = unsafe { OracleLibrary::open(&path)? };
    let table = Arc::new(recover_grid(&oracle)?);
    cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .insert(path, table.clone());
    Ok(table)
}

/// Compute a bounded reference directly from the verified libggml IQ1_S
/// dequantizer and the exact packed Q8_1 activation consumed by the CUDA MMQ
/// kernel. This is intentionally used only by the pre-timed persistent sample.
pub(crate) fn libggml_iq1s_reference_outputs(
    signature: &GgmlType19Signature,
    packed_matrix: &[u8],
    packed_activations: &[u8],
    path: Option<&Path>,
) -> Result<Vec<f32>, String> {
    signature.validate()?;
    if signature.ne11 != 1 || signature.stride11 != 1 {
        return Err("libggml IQ1_S reference requires one tightly packed activation lane".into());
    }
    if packed_matrix.len() != signature.matrix_storage_bytes()? {
        return Err("libggml IQ1_S reference matrix extent mismatch".into());
    }
    if packed_activations.len() != signature.activation_storage_bytes()? {
        return Err("libggml IQ1_S reference activation extent mismatch".into());
    }
    let selected = select_libggml_path(path)?;
    let path = selected
        .canonicalize()
        .map_err(|error| format!("libggml {}: {error}", selected.display()))?;
    let oracle = unsafe { OracleLibrary::open(&path)? };
    let columns = usize::try_from(signature.ne00)
        .map_err(|_| "libggml IQ1_S reference column count does not fit usize")?;
    let rows = usize::try_from(signature.ne01)
        .map_err(|_| "libggml IQ1_S reference row count does not fit usize")?;
    let columns_i64 = i64::try_from(signature.ne00)
        .map_err(|_| "libggml IQ1_S reference column count does not fit i64")?;
    let row_stride = usize::try_from(signature.stride01)
        .ok()
        .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
        .ok_or("libggml IQ1_S reference row stride overflow")?;
    let logical_row_bytes = columns
        .checked_div(IQ1S_BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
        .ok_or("libggml IQ1_S reference row extent overflow")?;
    if row_stride != logical_row_bytes {
        return Err("libggml IQ1_S reference does not allow padded matrix rows".into());
    }

    let mut activation = Vec::with_capacity(columns);
    for record in iter_q8_1_mmq(signature, packed_activations)? {
        for block in record?.subblocks {
            activation.extend(block.qs.map(|quant| block.d * f32::from(quant)));
        }
    }
    if activation.len() != columns || activation.iter().any(|value| !value.is_finite()) {
        return Err("libggml IQ1_S reference decoded an invalid activation".into());
    }

    let mut weights = vec![f32::NAN; columns];
    let mut outputs = Vec::with_capacity(rows);
    for row in 0..rows {
        let offset = row
            .checked_mul(row_stride)
            .ok_or("libggml IQ1_S reference row offset overflow")?;
        unsafe {
            (oracle.dequantize)(
                packed_matrix[offset..offset + logical_row_bytes]
                    .as_ptr()
                    .cast(),
                weights.as_mut_ptr().cast(),
                columns_i64,
            );
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "libggml IQ1_S reference produced a nonfinite weight in row {row}"
            ));
        }
        let value = weights
            .iter()
            .zip(&activation)
            .fold(0.0_f32, |sum, (&weight, &input)| {
                (sum + (weight * input) as f32) as f32
            });
        if !value.is_finite() {
            return Err(format!(
                "libggml IQ1_S reference produced a nonfinite output in row {row}"
            ));
        }
        outputs.push(value);
    }
    Ok(outputs)
}

fn cuda_mmq_iq1s_reference_outputs_with_grid(
    signature: &GgmlType19Signature,
    packed_matrix: &[u8],
    packed_activations: &[u8],
    grid: &GridTable,
) -> Result<Vec<f32>, String> {
    signature.validate()?;
    if signature.ne11 != 1 || signature.stride11 != 1 {
        return Err("CUDA MMQ IQ1_S reference requires one tightly packed activation lane".into());
    }
    if packed_matrix.len() != signature.matrix_storage_bytes()? {
        return Err("CUDA MMQ IQ1_S reference matrix extent mismatch".into());
    }
    if packed_activations.len() != signature.activation_storage_bytes()? {
        return Err("CUDA MMQ IQ1_S reference activation extent mismatch".into());
    }
    let columns = usize::try_from(signature.ne00)
        .map_err(|_| "CUDA MMQ IQ1_S column count does not fit usize")?;
    let rows = usize::try_from(signature.ne01)
        .map_err(|_| "CUDA MMQ IQ1_S row count does not fit usize")?;
    let row_stride = usize::try_from(signature.stride01)
        .ok()
        .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
        .ok_or("CUDA MMQ IQ1_S row stride overflow")?;
    let blocks_per_row = columns / IQ1S_BLOCK_VALUES;
    if row_stride != blocks_per_row * IQ1S_BLOCK_BYTES {
        return Err("CUDA MMQ IQ1_S reference does not allow padded matrix rows".into());
    }
    let mut q8_blocks = Vec::with_capacity(columns / GROUP_VALUES);
    for record in iter_q8_1_mmq(signature, packed_activations)? {
        q8_blocks.extend(record?.subblocks);
    }
    if q8_blocks.len() != columns / GROUP_VALUES {
        return Err("CUDA MMQ IQ1_S activation group count mismatch".into());
    }

    let mut outputs = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_offset = row
            .checked_mul(row_stride)
            .ok_or("CUDA MMQ IQ1_S row offset overflow")?;
        let row_bytes = &packed_matrix[row_offset..row_offset + row_stride];
        let mut sum = 0.0f32;
        for block_index in 0..blocks_per_row {
            let block_offset = block_index * IQ1S_BLOCK_BYTES;
            let block = Iq1sBlock::parse(
                &row_bytes[block_offset..block_offset + IQ1S_BLOCK_BYTES],
                grid,
            )?;
            for (group_index, group) in block.groups.iter().enumerate() {
                let q8 = q8_blocks
                    .get(block_index * block.groups.len() + group_index)
                    .ok_or("CUDA MMQ IQ1_S Q8 group is missing")?;
                let (grid_dot, _) = raw_component_dots(group, q8);
                let q8_sum = q8.qs.iter().map(|&value| i64::from(value)).sum::<i64>();
                let encoded_grid_dot = grid_dot
                    .checked_add(q8_sum)
                    .ok_or("CUDA MMQ IQ1_S encoded grid dot overflow")?;
                let d1q = (block.d * f32::from(group.odd_scale)) as f32;
                let weight_d = round_to_half(d1q)?;
                let delta = (-1.0f32 + f32::from(group.delta_sign) * 0.125) as f32;
                let weight_m = round_to_half((d1q * delta) as f32)?;
                let product_d = round_to_half((weight_d * q8.d) as f32)?;
                let product_m = round_to_half((weight_m * q8.s) as f32)?;
                let contribution = (encoded_grid_dot as f32).mul_add(product_d, product_m);
                sum = (sum + contribution) as f32;
            }
        }
        if !sum.is_finite() {
            return Err(format!(
                "CUDA MMQ IQ1_S reference produced a nonfinite output in row {row}"
            ));
        }
        outputs.push(sum);
    }
    Ok(outputs)
}

pub(crate) fn cuda_mmq_iq1s_reference_outputs(
    signature: &GgmlType19Signature,
    packed_matrix: &[u8],
    packed_activations: &[u8],
    path: Option<&Path>,
) -> Result<Vec<f32>, String> {
    let grid = validated_grid(path)?;
    cuda_mmq_iq1s_reference_outputs_with_grid(signature, packed_matrix, packed_activations, &grid)
}

impl OracleLibrary {
    unsafe fn open(path: &Path) -> Result<Self, String> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| "libggml path contains NUL")?;
        let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            return Err(format!("dlopen libggml failed: {}", dlerror()));
        }
        let result = (|| {
            let dequantize = symbol::<DequantizeIq1s>(handle, b"dequantize_row_iq1_s\0")?;
            let init = symbol::<GgmlInit>(handle, b"ggml_init\0")?;
            let free = symbol::<GgmlFree>(handle, b"ggml_free\0")?;
            let context = init(GgmlInitParams {
                mem_size: 0,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            if context.is_null() {
                return Err("ggml_init failed".into());
            }
            free(context);
            Ok(Self { handle, dequantize })
        })();
        if result.is_err() {
            libc::dlclose(handle);
        }
        result
    }
}

unsafe fn symbol<T: Copy>(handle: *mut libc::c_void, name: &'static [u8]) -> Result<T, String> {
    let pointer = libc::dlsym(handle, name.as_ptr().cast());
    if pointer.is_null() {
        Err(format!(
            "libggml missing symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        ))
    } else {
        Ok(std::mem::transmute_copy(&pointer))
    }
}

unsafe fn dlerror() -> String {
    let error = libc::dlerror();
    if error.is_null() {
        "unknown loader error".into()
    } else {
        CStr::from_ptr(error).to_string_lossy().into_owned()
    }
}

fn recover_grid(oracle: &OracleLibrary) -> Result<GridTable, String> {
    let mut table = [[0_i8; 8]; GRID_ENTRIES];
    let mut packed = [0_u8; IQ1S_BLOCK_BYTES];
    packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut output = [0_f32; IQ1S_BLOCK_VALUES];
    for (index, entry) in table.iter_mut().enumerate() {
        packed[2..].fill(0);
        packed[2] = index as u8;
        packed[34..36].copy_from_slice(&((index >> 8) as u16).to_le_bytes());
        output.fill(f32::NAN);
        unsafe {
            (oracle.dequantize)(
                packed.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                IQ1S_BLOCK_VALUES as i64,
            );
        }
        for (dst, &value) in entry.iter_mut().zip(&output[..8]) {
            let candidate = value - 0.125;
            if !candidate.is_finite() || !matches!(candidate, -1.0 | 0.0 | 1.0) {
                return Err(format!(
                    "libggml oracle returned invalid grid value for index {index}"
                ));
            }
            *dst = candidate as i8;
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::cxl_tmatmul_v3::{
        BufferV3, CapsV3, CommitV3, CompletionV3, SubmitV3, WaitV3,
    };
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CAPTURE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn capture_test_guard() -> std::sync::MutexGuard<'static, ()> {
        CAPTURE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct RemovedEnvGuard {
        name: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl RemovedEnvGuard {
        fn new(name: &'static str) -> Self {
            let original = std::env::var_os(name);
            std::env::remove_var(name);
            Self { name, original }
        }
    }

    impl Drop for RemovedEnvGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    #[test]
    fn strict_qwen_requires_an_explicit_libggml_path() {
        let _env_lock = crate::r#impl::test_env::lock();
        let _library = RemovedEnvGuard::new("HETGPU_LIBGGML");
        let original_strict = std::env::var_os("HETGPU_QWEN_IQ1S_STRICT");
        std::env::set_var("HETGPU_QWEN_IQ1S_STRICT", "1");

        let result = select_libggml_path(None);

        match original_strict {
            Some(value) => std::env::set_var("HETGPU_QWEN_IQ1S_STRICT", value),
            None => std::env::remove_var("HETGPU_QWEN_IQ1S_STRICT"),
        }
        assert_eq!(
            result.unwrap_err(),
            "strict Qwen IQ1_S mode requires verified HETGPU_LIBGGML"
        );
    }

    #[test]
    fn libggml_reference_dequantizes_exact_packed_iq1s_rows() {
        let qwen_libggml = Path::new("/root/qwen35-au250-build/llama-build/bin/libggml.so.0.22.0");
        if !qwen_libggml.is_file() {
            return;
        }
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
        let matrix = vec![0u8; IQ1S_BLOCK_BYTES];
        let activations = vec![0u8; 2 * Q8_1_MMQ_BYTES];
        let outputs =
            libggml_iq1s_reference_outputs(&signature, &matrix, &activations, Some(qwen_libggml))
                .unwrap();
        assert_eq!(outputs, vec![0.0]);
    }

    #[test]
    fn cuda_mmq_reference_uses_blackwell_half2_product_rounding() {
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
        let mut matrix = vec![0u8; IQ1S_BLOCK_BYTES];
        matrix[..2].copy_from_slice(&0x3555u16.to_le_bytes());
        matrix[34..36].copy_from_slice(&(7u16 << 12).to_le_bytes());
        let mut activation = vec![0u8; 2 * Q8_1_MMQ_BYTES];
        activation[2..4].copy_from_slice(&0x3555u16.to_le_bytes());

        let output =
            cuda_mmq_iq1s_reference_outputs_with_grid(&signature, &matrix, &activation, &grid())
                .expect("evaluate CUDA MMQ reference");
        let d1q = half_to_f32(0x3555) * 15.0;
        let delta_weight = round_to_half(d1q * -0.875).unwrap();
        let expected = round_to_half(delta_weight * half_to_f32(0x3555)).unwrap();
        assert_eq!(output, vec![expected]);
        assert_ne!(expected, delta_weight * half_to_f32(0x3555));
    }

    fn grid() -> GridTable {
        let mut entries = [[0_i8; 8]; GRID_ENTRIES];
        for (index, entry) in entries.iter_mut().enumerate() {
            for (column, value) in entry.iter_mut().enumerate() {
                *value = match (index + column) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
            }
        }
        entries
    }

    fn block(odd: u8, negative: bool, high_index: u16) -> [u8; IQ1S_BLOCK_BYTES] {
        let mut packed = [0_u8; IQ1S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // half(1)
        packed[2] = high_index as u8;
        let qh = ((u16::from(odd) - 1) / 2) << 12
            | if negative { 0x8000 } else { 0 }
            | ((high_index >> 8) & 7);
        packed[34..36].copy_from_slice(&qh.to_le_bytes());
        packed
    }

    #[derive(Serialize)]
    struct RtlGroupVector {
        group: usize,
        odd_scale: u8,
        delta_sign: i8,
        grid_indices: [usize; 4],
        grid_values: [i8; GROUP_VALUES],
        q8_d_bits: u32,
        q8_s_bits: u32,
        q8_values: [i8; GROUP_VALUES],
        grid_dot: i64,
        delta_dot: i64,
        reconstructed_f32_bits: u32,
    }

    #[derive(Serialize)]
    struct RtlBlockVector {
        name: String,
        block_hex: String,
        iq1s_d_bits: u32,
        groups: Vec<RtlGroupVector>,
    }

    #[derive(Serialize)]
    struct RtlVectorFile {
        format: &'static str,
        grid_entries: usize,
        grid_sha256: String,
        vectors: Vec<RtlBlockVector>,
    }

    fn iq1s_grid_memh(grid: &GridTable) -> String {
        let mut output = String::with_capacity(GRID_ENTRIES * 17);
        for entry in grid {
            // $readmemh places the rightmost byte in bits [7:0].
            for value in entry.iter().rev() {
                write!(&mut output, "{:02x}", *value as u8).unwrap();
            }
            output.push('\n');
        }
        output
    }

    fn build_rtl_vectors(grid: &GridTable) -> RtlVectorFile {
        let mut vectors = Vec::with_capacity(16);
        for vector_index in 0..16_usize {
            let mut packed = [0_u8; IQ1S_BLOCK_BYTES];
            let iq1s_d_bits: u16 = if vector_index % 2 == 0 {
                0x3555
            } else {
                0xb400
            };
            packed[..2].copy_from_slice(&iq1s_d_bits.to_le_bytes());
            let mut expected = Vec::with_capacity(8);
            for group_index in 0..8_usize {
                let odd_scale = 1 + 2 * ((vector_index + group_index) % 8) as u8;
                let delta_sign = if (vector_index + group_index) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let indices = [
                    (vector_index * 137 + group_index * 29) & 0x7ff,
                    0x7ff - ((vector_index * 73 + group_index * 11) & 0x7ff),
                    ((vector_index << 8) | (group_index * 31)) & 0x7ff,
                    ((7 - group_index) << 8) | (255 - vector_index * 13) & 0xff,
                ];
                let mut qh = (u16::from((odd_scale - 1) / 2) << 12)
                    | if delta_sign < 0 { 0x8000 } else { 0 };
                for (position, index) in indices.iter().enumerate() {
                    packed[2 + group_index * 4 + position] = *index as u8;
                    qh |= (((index >> 8) & 7) as u16) << (3 * position);
                }
                let qh_offset = 34 + group_index * 2;
                packed[qh_offset..qh_offset + 2].copy_from_slice(&qh.to_le_bytes());

                let mut qs = [0_i8; GROUP_VALUES];
                for (position, quant) in qs.iter_mut().enumerate() {
                    *quant = match vector_index {
                        0 => 0,
                        1 => {
                            if position % 2 == 0 {
                                i8::MIN
                            } else {
                                i8::MAX
                            }
                        }
                        _ => {
                            (((position * 37 + group_index * 19 + vector_index * 11) % 255) as i16
                                - 127) as i8
                        }
                    };
                }
                let q8 = Q8_1Block {
                    d: [0.5_f32, -0.25, 1.5, 0.0625][(vector_index + group_index) % 4],
                    s: [-2.0_f32, 0.0, 0.75, 4.0][(vector_index * 3 + group_index) % 4],
                    qs,
                };
                let mut values = [0_i8; GROUP_VALUES];
                for position in 0..4 {
                    values[position * 8..(position + 1) * 8]
                        .copy_from_slice(&grid[indices[position]]);
                }
                let group = Iq1sGroup {
                    odd_scale,
                    delta_sign,
                    grid_indices: indices,
                    grid_values: values,
                };
                let (grid_dot, delta_dot) = raw_component_dots(&group, &q8);
                let reconstructed = reconstruct_from_raw(
                    &group,
                    half_to_f32(iq1s_d_bits),
                    &q8,
                    grid_dot << 8,
                    delta_dot << 8,
                )
                .unwrap();
                expected.push(RtlGroupVector {
                    group: group_index,
                    odd_scale,
                    delta_sign,
                    grid_indices: indices,
                    grid_values: values,
                    q8_d_bits: q8.d.to_bits(),
                    q8_s_bits: q8.s.to_bits(),
                    q8_values: qs,
                    grid_dot,
                    delta_dot,
                    reconstructed_f32_bits: reconstructed.to_bits(),
                });
            }
            let parsed = Iq1sBlock::parse(&packed, grid).unwrap();
            for (actual, expected_group) in parsed.groups.iter().zip(&expected) {
                assert_eq!(actual.odd_scale, expected_group.odd_scale);
                assert_eq!(actual.delta_sign, expected_group.delta_sign);
                assert_eq!(actual.grid_indices, expected_group.grid_indices);
                assert_eq!(actual.grid_values, expected_group.grid_values);
            }
            vectors.push(RtlBlockVector {
                name: format!("iq1s_{vector_index:02}"),
                block_hex: hex(&packed),
                iq1s_d_bits: half_to_f32(iq1s_d_bits).to_bits(),
                groups: expected,
            });
        }
        let grid_memh = iq1s_grid_memh(grid);
        RtlVectorFile {
            format: "hetgpu-iq1s-rtl-v1",
            grid_entries: GRID_ENTRIES,
            grid_sha256: hex(&Sha256::digest(grid_memh.as_bytes())),
            vectors,
        }
    }

    #[test]
    #[ignore = "writes deterministic RTL fixtures when explicit output paths are set"]
    fn emit_iq1s_rtl_vectors() {
        let _env_lock = crate::r#impl::test_env::lock();
        let grid = validated_grid(None).expect("recover the canonical IQ1_S grid");
        assert!(grid
            .iter()
            .flatten()
            .all(|value| matches!(value, -1 | 0 | 1)));
        let grid_memh = iq1s_grid_memh(&grid);
        let vectors = build_rtl_vectors(&grid);
        assert_eq!(vectors.vectors.len(), 16);
        assert!(vectors
            .vectors
            .iter()
            .all(|vector| vector.groups.len() == 8));
        assert_eq!(
            vectors.grid_sha256,
            hex(&Sha256::digest(grid_memh.as_bytes()))
        );

        if let Some(path) = std::env::var_os("HETGPU_IQ1S_GRID_OUT") {
            std::fs::write(path, &grid_memh).expect("write IQ1_S grid memh");
        }
        if let Some(path) = std::env::var_os("HETGPU_IQ1S_GRID_SHA256_OUT") {
            std::fs::write(path, format!("{}\n", vectors.grid_sha256))
                .expect("write IQ1_S grid checksum");
        }
        if let Some(path) = std::env::var_os("HETGPU_RTL_VECTOR_OUT") {
            let json = serde_json::to_vec_pretty(&vectors).expect("serialize IQ1_S RTL vectors");
            std::fs::write(path, json).expect("write IQ1_S RTL vectors");
        }
    }

    fn kimi_signature() -> GgmlType19Signature {
        GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 7168,
            ne01: 2048,
            stride01: 28,
            ne10: 7168,
            ne11: 1,
            stride11: 8064,
            ne0: 2048,
        }
    }

    struct CountingVecCopies {
        matrix_reads: AtomicUsize,
        activation_reads: AtomicUsize,
        activation: Vec<u8>,
    }

    impl VecCaptureCopies for CountingVecCopies {
        fn copy_matrix(&self, _pointer: usize, bytes: usize) -> Result<Vec<u8>, String> {
            self.matrix_reads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0; bytes])
        }

        fn copy_activation(&self, _pointer: usize, bytes: usize) -> Result<Vec<u8>, String> {
            self.activation_reads.fetch_add(1, Ordering::SeqCst);
            if bytes != self.activation.len() {
                return Err(format!(
                    "activation copy requested {bytes} bytes, fixture has {}",
                    self.activation.len()
                ));
            }
            Ok(self.activation.clone())
        }
    }

    #[test]
    fn persistent_vec_capture_copies_activation_without_matrix() {
        let signature = GgmlType19VecSignature {
            kernel: "mul_mat_vec_q_ggml_type19".to_string(),
            ncols_x: 4096,
            nrows_x: 1024,
            nrows_y: 4096,
            nrows_dst: 1024,
        };
        let copies = CountingVecCopies {
            matrix_reads: AtomicUsize::new(0),
            activation_reads: AtomicUsize::new(0),
            activation: vec![0; 4096 / 32 * Q8_1_BLOCK_BYTES],
        };

        let capture =
            capture_vec_activation_with(0x1000, 0x2000, 0x3000, 7, [0x55; 32], signature, &copies)
                .unwrap();

        assert_eq!(copies.matrix_reads.load(Ordering::SeqCst), 0);
        assert_eq!(copies.activation_reads.load(Ordering::SeqCst), 1);
        assert_eq!(capture.packed_activations().len(), 4096 / 128 * 144);
        assert_eq!(capture.launch.matrix_ptr, 0x1000);
    }

    #[test]
    fn persistent_vec_capture_rejects_zero_hash_before_activation_copy() {
        let signature = GgmlType19VecSignature {
            kernel: "mul_mat_vec_q_ggml_type19".to_string(),
            ncols_x: 4096,
            nrows_x: 1024,
            nrows_y: 4096,
            nrows_dst: 1024,
        };
        let copies = CountingVecCopies {
            matrix_reads: AtomicUsize::new(0),
            activation_reads: AtomicUsize::new(0),
            activation: vec![0; 4096 / 32 * Q8_1_BLOCK_BYTES],
        };

        let error =
            capture_vec_activation_with(0x1000, 0x2000, 0x3000, 7, [0; 32], signature, &copies)
                .unwrap_err();

        assert!(error.contains("nonzero content hash"), "{error}");
        assert_eq!(copies.matrix_reads.load(Ordering::SeqCst), 0);
        assert_eq!(copies.activation_reads.load(Ordering::SeqCst), 0);
    }

    fn batched_signature(batch: u64, stride11: u64) -> GgmlType19Signature {
        GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: batch,
            stride11,
            ne0: 1,
        }
    }

    #[test]
    fn accepts_producer_pitch_and_computes_last_active_record_extent() {
        let signature = batched_signature(2, 3);
        assert_eq!(signature.validate().unwrap(), signature);
        assert_eq!(
            signature.activation_storage_bytes().unwrap(),
            5 * Q8_1_MMQ_BYTES
        );

        let mut too_narrow = signature;
        too_narrow.stride11 = 1;
        assert!(too_narrow.validate().unwrap_err().contains("stride11"));
    }

    #[test]
    fn activation_extent_checks_pitch_boundaries_and_overflow() {
        let batch_one = batched_signature(1, 1);
        assert_eq!(
            batch_one.activation_storage_bytes().unwrap(),
            2 * Q8_1_MMQ_BYTES
        );

        let overflow = batched_signature(2, u64::MAX);
        assert!(overflow
            .activation_storage_bytes()
            .unwrap_err()
            .contains("overflow"));
    }

    fn logical_launch_for_validation(signature: GgmlType19Signature) -> LogicalLaunch {
        LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [0x5a; 32],
            signature,
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn validate_before_copy_rejects_batch_outside_scheduler_u32_domain() {
        let batch = u64::from(u32::MAX) + 1;
        let launch = logical_launch_for_validation(batched_signature(batch, batch));

        let error = launch.validate_before_copy().unwrap_err();

        assert!(error.contains("u32 scheduler batch domain"), "{error}");
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn validate_before_copy_rejects_full_output_extent_overflow() {
        let mut signature = batched_signature(u64::from(u32::MAX), u64::from(u32::MAX));
        signature.ne01 = u64::from(u32::MAX);
        signature.ne0 = u64::from(u32::MAX);
        let launch = logical_launch_for_validation(signature);

        let error = launch.validate_before_copy().unwrap_err();

        assert!(error.contains("output extent overflow"), "{error}");
    }

    #[test]
    fn execution_descriptor_count_accepts_software_ceiling_boundary() {
        assert_eq!(
            checked_execution_descriptor_count(MAX_EXECUTION_DESCRIPTORS, 1).unwrap(),
            MAX_EXECUTION_DESCRIPTORS
        );
    }

    #[test]
    fn execution_descriptor_count_rejects_over_limit_and_arithmetic_overflow() {
        let over_limit =
            checked_execution_descriptor_count(MAX_EXECUTION_DESCRIPTORS + 1, 1).unwrap_err();
        assert!(over_limit.contains("software safety limit"), "{over_limit}");

        let overflow = checked_execution_descriptor_count(usize::MAX, 2).unwrap_err();
        assert!(overflow.contains("overflow"), "{overflow}");
    }

    fn producer_pitched_activations() -> Vec<u8> {
        let mut activations = vec![0_u8; 5 * Q8_1_MMQ_BYTES];
        for (record_index, marker) in [(0, 10), (1, 11), (3, 30), (4, 31)] {
            activations[record_index * Q8_1_MMQ_BYTES + 16] = marker;
        }
        activations
    }

    #[test]
    fn q8_group_selects_k_record_then_batch_and_rejects_out_of_range_indices() {
        let _capture_guard = capture_test_guard();
        let signature = batched_signature(2, 3);
        let launch = LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [3; 32],
            signature,
        };
        let matrix = block(1, false, 0);
        let activations = producer_pitched_activations();

        let captured = capture_from_host(launch, &matrix, &activations, &grid()).unwrap();
        assert_eq!(captured.q8_group(0, 0).unwrap().qs[0], 10);
        assert_eq!(captured.q8_group(1, 0).unwrap().qs[0], 11);
        assert_eq!(captured.q8_group(0, 4).unwrap().qs[0], 30);
        assert_eq!(captured.q8_group(1, 4).unwrap().qs[0], 31);
        assert!(captured.q8_group(2, 0).unwrap_err().contains("batch"));
        assert!(captured.q8_group(0, 8).unwrap_err().contains("group"));
    }

    #[test]
    fn capture_parses_active_records_and_ignores_pitch_padding() {
        let _capture_guard = capture_test_guard();
        let signature = batched_signature(2, 3);
        let launch = LogicalLaunch {
            matrix_ptr: 0x4000,
            activation_ptr: 0x5000,
            output_ptr: 0x6000,
            allocation_generation: 1,
            content_hash: [4; 32],
            signature,
        };
        let matrix = block(1, false, 0);
        let mut activations = producer_pitched_activations();
        let padding_offset = 2 * Q8_1_MMQ_BYTES;
        activations[padding_offset..padding_offset + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        let captured = capture_from_host(launch.clone(), &matrix, &activations, &grid()).unwrap();
        assert_eq!(captured.activation_blocks().unwrap().count(), 4);

        let active_offset = 3 * Q8_1_MMQ_BYTES;
        activations[active_offset..active_offset + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        assert!(capture_from_host(launch, &matrix, &activations, &grid())
            .unwrap_err()
            .contains("finite"));
    }

    #[test]
    fn validates_real_kimi_block_unit_signature() {
        let signature = kimi_signature();
        assert_eq!(signature.validate().unwrap(), signature);
        assert_eq!(
            signature.activation_storage_bytes().unwrap(),
            (55 * 8064 + 1) * Q8_1_MMQ_BYTES
        );
    }

    #[test]
    fn decomposes_all_odd_scales_both_signs_and_high_grid_bits() {
        let grid = grid();
        for odd in [1_u8, 3, 5, 7, 9, 11, 13, 15] {
            for negative in [false, true] {
                let parsed = Iq1sBlock::parse(&block(odd, negative, 0x7ff), &grid).unwrap();
                assert_eq!(parsed.groups[0].odd_scale, odd);
                assert_eq!(parsed.groups[0].delta_sign, if negative { -1 } else { 1 });
                assert_eq!(parsed.groups[0].grid_indices[0], 0x7ff);
                assert_eq!(parsed.groups[0].grid_values[..8], grid[0x7ff]);
            }
        }
    }

    #[test]
    fn parses_exact_144_byte_ds4_and_rejects_bad_storage() {
        let mut packed = [0_u8; Q8_1_MMQ_BYTES];
        for pair in 0..4 {
            packed[pair * 4..pair * 4 + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
            packed[pair * 4 + 2..pair * 4 + 4].copy_from_slice(&0x3e00_u16.to_le_bytes());
            packed[16 + pair * 32..16 + (pair + 1) * 32].fill((pair + 1) as u8);
        }
        let parsed = Q8_1MmqBlock::parse(&packed).unwrap();
        assert_eq!((parsed.subblocks[0].d, parsed.subblocks[0].s), (0.5, 1.5));
        assert_eq!(parsed.subblocks[3].qs, [4; 32]);
        assert!(Q8_1MmqBlock::parse(&packed[..143]).is_err());
        packed[0..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        assert!(Q8_1MmqBlock::parse(&packed).unwrap_err().contains("finite"));
    }

    #[test]
    fn canonicalizes_one_token_q8_vector_into_ds4_groups() {
        let signature = GgmlType19VecSignature {
            kernel: "_Z13mul_mat_vec_qIL9ggml_type19ELi1EEvPKvS2_Pfiiii".into(),
            ncols_x: 256,
            nrows_x: 256,
            nrows_y: 256,
            nrows_dst: 256,
        };
        let mut raw = vec![0_u8; signature.activation_storage_bytes().unwrap()];
        for block_index in 0..8 {
            let offset = block_index * Q8_1_BLOCK_BYTES;
            raw[offset..offset + 4].copy_from_slice(&[
                block_index as u8,
                block_index as u8 + 1,
                block_index as u8 + 2,
                block_index as u8 + 3,
            ]);
            raw[offset + 4..offset + Q8_1_BLOCK_BYTES].fill(block_index as u8 + 10);
        }

        let canonical = canonicalize_q8_1_vector(&signature, &raw).unwrap();
        assert_eq!(canonical.len(), 2 * Q8_1_MMQ_BYTES);
        assert_eq!(
            &canonical[..16],
            &raw[..4 * Q8_1_BLOCK_BYTES]
                .chunks(36)
                .flat_map(|block| block[..4].iter().copied())
                .collect::<Vec<_>>()
        );
        assert_eq!(&canonical[16..48], &[10; 32]);
        assert_eq!(&canonical[48..80], &[11; 32]);
        assert_eq!(&canonical[80..112], &[12; 32]);
        assert_eq!(&canonical[112..144], &[13; 32]);
        assert_eq!(signature.mmq_signature().unwrap().ne11, 1);
        assert_eq!(signature.mmq_signature().unwrap().stride11, 1);
    }

    #[test]
    fn real_kimi_geometry_has_four_edge_tiles_and_compact_groups() {
        let tiles = plan_tiles(&kimi_signature()).unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!((tiles[3].valid_out, tiles[3].valid_in), (2048, 1024));
        assert_eq!(
            tiles.iter().map(|tile| tile.group_count).sum::<usize>(),
            224
        );

        let parsed = Iq1sBlock::parse(&block(15, true, 0x7ff), &grid()).unwrap();
        let component = ComponentTile::from_rows(
            TileGeometry {
                row_tile: 0,
                column_tile: 0,
                valid_out: 1,
                valid_in: 256,
                group_count: 8,
            },
            0,
            ComponentKind::Delta,
            &[parsed],
        )
        .unwrap();
        assert_eq!(component.values.len(), 32);
        assert_eq!(component.values, vec![-1; 32]);
        let full = component.pack_ternary2().unwrap();
        assert_eq!(full.len(), TILE_PACKED_BYTES);
        assert!(full[8..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn validates_both_signed_raw_components_and_mmq_half_boundary() {
        let group = Iq1sGroup {
            odd_scale: 15,
            delta_sign: -1,
            grid_indices: [0; 4],
            grid_values: [
                1, 0, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, -1, 0, 1, -1, 1,
                0, -1, 1, -1, 0, 1, -1,
            ],
        };
        let mut qs = [0_i8; 32];
        for (index, quant) in qs.iter_mut().enumerate() {
            *quant = index as i8 - 16;
        }
        let q8 = Q8_1Block {
            d: -0.1875,
            s: 1.5,
            qs,
        };
        let (grid_dot, delta_dot) = raw_component_dots(&group, &q8);
        assert_eq!(
            validate_raw_q8_8(grid_dot << 8, grid_dot).unwrap(),
            grid_dot
        );
        assert_eq!(
            validate_raw_q8_8(delta_dot << 8, delta_dot).unwrap(),
            delta_dot
        );
        assert!(validate_raw_q8_8((grid_dot << 8) + 1, grid_dot).is_err());
        let result =
            reconstruct_from_raw(&group, 0.333251953125, &q8, grid_dot << 8, delta_dot << 8)
                .unwrap();
        assert_eq!(result.to_bits(), 0x41ac8000);
    }

    #[test]
    fn cache_identity_changes_on_every_bound_field() {
        let base = MatrixCacheIdentity {
            matrix_ptr: 0x1000,
            signature: kimi_signature(),
            allocation_generation: 7,
            content_hash: [0x55; 32],
        };
        let mut cache = ComponentCache::default();
        assert!(!cache.matches(&base));
        let source_signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let payload = MatrixSource::new(
            source_signature,
            Arc::from(block(1, false, 0)),
            Arc::new(grid()),
        )
        .unwrap();
        cache.insert(base.clone(), payload.clone());
        assert!(cache.matches(&base));
        assert!(Arc::ptr_eq(&cache.get(&base).unwrap(), &payload));
        let mut changed = base.clone();
        changed.matrix_ptr += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base.clone();
        changed.signature.ne0 += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base.clone();
        changed.allocation_generation += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base;
        changed.content_hash[0] ^= 1;
        assert!(!cache.matches(&changed));
    }

    #[test]
    fn component_cache_retains_distinct_live_matrix_identities() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let first_identity = MatrixCacheIdentity {
            matrix_ptr: 0x1000,
            signature: signature.clone(),
            allocation_generation: 1,
            content_hash: [0x11; 32],
        };
        let second_identity = MatrixCacheIdentity {
            matrix_ptr: 0x2000,
            signature: signature.clone(),
            allocation_generation: 1,
            content_hash: [0x22; 32],
        };
        let first = MatrixSource::new(
            signature.clone(),
            Arc::from(block(1, false, 0)),
            Arc::new(grid()),
        )
        .unwrap();
        let second =
            MatrixSource::new(signature, Arc::from(block(3, true, 1)), Arc::new(grid())).unwrap();

        let mut cache = ComponentCache::default();
        cache.insert(first_identity.clone(), first.clone());
        cache.insert(second_identity.clone(), second.clone());

        assert!(Arc::ptr_eq(&cache.get(&first_identity).unwrap(), &first));
        assert!(Arc::ptr_eq(&cache.get(&second_identity).unwrap(), &second));
    }

    #[test]
    fn fixture_json_contains_packed_inputs_raw_parts_and_f32_bits() {
        let json = fixture_json(&[1, 2], &[3, 4], &[(256, -512)], &[1.5]).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["iq1s_hex"], "0102");
        assert_eq!(value["q8_1_mmq_hex"], "0304");
        assert_eq!(value["raw_components"][0][1], -512);
        assert_eq!(value["outputs_f32_bits"][0], 1.5_f32.to_bits());
    }

    #[test]
    fn matrix_decomposition_selects_each_physical_iq1s_block_without_dense_float() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 2304,
            ne01: 1,
            stride01: 9,
            ne10: 2304,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let mut matrix = Vec::new();
        for index in 0..9_u16 {
            matrix.extend_from_slice(&block(1, index == 1, index));
        }
        let components = decompose_component_tiles(&signature, &matrix, &grid()).unwrap();
        assert_eq!(components.len(), 144);
        let selected = components
            .iter()
            .find(|tile| {
                tile.geometry.column_tile == 0
                    && tile.group32 == 8
                    && tile.kind == ComponentKind::Delta
            })
            .unwrap();
        assert_eq!(selected.values, vec![-1; 32]);
        let edge = components.last().unwrap();
        assert_eq!(
            (edge.geometry.column_tile, edge.geometry.valid_in),
            (1, 256)
        );
        assert!(
            decompose_component_tiles(&signature, &matrix[..matrix.len() - 1], &grid()).is_err()
        );
    }

    #[test]
    fn host_capture_validates_ds4_and_reuses_only_the_full_matrix_identity() {
        let _capture_guard = capture_test_guard();
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [7; 32],
            signature,
        };
        let matrix = block(3, false, 0x700);
        let activations = vec![0_u8; 2 * Q8_1_MMQ_BYTES];
        let first = capture_from_host(launch.clone(), &matrix, &activations, &grid()).unwrap();
        let second = capture_from_host(launch.clone(), &matrix, &activations, &grid()).unwrap();
        assert!(Arc::ptr_eq(&first.matrix, &second.matrix));
        assert_eq!(first.activation_blocks().unwrap().count(), 2);

        let mut changed = launch;
        changed.content_hash[0] ^= 1;
        let third = capture_from_host(changed, &matrix, &activations, &grid()).unwrap();
        assert!(!Arc::ptr_eq(&first.matrix, &third.matrix));
        assert!(
            capture_from_host(first.launch.clone(), &matrix, &activations[..287], &grid()).is_err()
        );
    }

    #[test]
    fn component_iteration_is_lazy_and_host_bounded() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 2304,
            ne01: 2050,
            stride01: 9,
            ne10: 2304,
            ne11: 1,
            stride11: 1,
            ne0: 2050,
        };
        let matrix = vec![0_u8; signature.matrix_storage_bytes().unwrap()];
        let source = MatrixSource::new(signature, Arc::from(matrix), Arc::new(grid())).unwrap();
        let mut iterator = source.component_iter().unwrap();
        assert!(std::mem::size_of_val(&iterator) <= 256);
        assert_eq!(iterator.generated_count(), 0);
        let first = iterator.next().unwrap().unwrap();
        assert_eq!(
            (
                first.geometry.row_tile,
                first.geometry.column_tile,
                first.group32,
                first.kind
            ),
            (0, 0, 0, ComponentKind::Grid)
        );
        assert_eq!(iterator.generated_count(), 1);
        assert!(iterator.remaining_count() > 1);
    }

    #[derive(Clone)]
    struct MemoryDax {
        bytes: Arc<Mutex<Vec<u8>>>,
        writes: Arc<Mutex<Vec<(u64, usize)>>>,
    }
    impl MemoryDax {
        fn new(bytes: usize) -> Self {
            Self {
                bytes: Arc::new(Mutex::new(vec![0; bytes])),
                writes: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[test]
    fn file_dax_access_uses_a_bounded_shared_mapping() {
        let file = tempfile::NamedTempFile::new().expect("create mapped DAX fixture");
        file.as_file()
            .set_len(4096)
            .expect("size mapped DAX fixture");
        let bound = BoundDaxFile::test_only(
            file.as_file().try_clone().expect("clone DAX fixture fd"),
            4096,
        );
        let dax = FileDaxAccess::map(&bound).expect("map bound DAX fixture");

        assert_eq!(dax.write(64, &[1, 2, 3, 4]).unwrap(), 4);
        let mut readback = [0_u8; 4];
        assert_eq!(dax.read(64, &mut readback).unwrap(), 4);
        assert_eq!(readback, [1, 2, 3, 4]);
        assert!(dax.write(4094, &[1, 2, 3]).unwrap_err().contains("range"));
        assert!(dax
            .read(4096, &mut [0_u8; 1])
            .unwrap_err()
            .contains("range"));

        drop(dax);
        let bytes = std::fs::read(file.path()).expect("read mapped DAX fixture backing file");
        assert_eq!(&bytes[64..68], &[1, 2, 3, 4]);
    }

    #[test]
    fn v3_memory_backend_parser_is_explicit_and_fail_closed() {
        validate_v3_memory_setting(None).unwrap();
        validate_v3_memory_setting(Some("dax")).unwrap();
        assert!(validate_v3_memory_setting(Some("ioctl")).is_err());
        assert!(validate_v3_memory_setting(Some("inline")).is_err());
        assert!(validate_v3_memory_setting(Some("mmap")).is_err());
    }

    impl DaxAccess for MemoryDax {
        fn write(&self, offset: u64, bytes: &[u8]) -> Result<usize, String> {
            let offset = offset as usize;
            self.bytes.lock().unwrap()[offset..offset + bytes.len()].copy_from_slice(bytes);
            self.writes
                .lock()
                .unwrap()
                .push((offset as u64, bytes.len()));
            Ok(bytes.len())
        }
        fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<usize, String> {
            let offset = offset as usize;
            bytes.copy_from_slice(&self.bytes.lock().unwrap()[offset..offset + bytes.len()]);
            Ok(bytes.len())
        }
    }

    struct FakeV3 {
        dax: MemoryDax,
        buffers: HashMap<u32, (u64, u64, u32, u64)>,
        next_handle: u32,
        pending: Vec<TaskV3>,
        submission: u64,
        max_batch: u32,
        submitted: Arc<Mutex<Vec<TaskV3>>>,
    }
    impl FakeV3 {
        fn new(dax: MemoryDax) -> Self {
            Self::with_max_batch(dax, 1).0
        }

        fn with_max_batch(dax: MemoryDax, max_batch: u32) -> (Self, Arc<Mutex<Vec<TaskV3>>>) {
            let submitted = Arc::new(Mutex::new(Vec::new()));
            let io = Self {
                dax,
                buffers: HashMap::new(),
                next_handle: 1,
                pending: Vec::new(),
                submission: 0,
                max_batch,
                submitted: submitted.clone(),
            };
            (io, submitted)
        }
    }
    impl IoctlOps for FakeV3 {
        fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
            caps.version = 3;
            caps.capabilities = 0xfb;
            caps.num_instances = MAX_LANES;
            caps.dim_d = 2048;
            caps.max_batch = self.max_batch;
            caps.max_descriptors = 1024;
            caps.max_inflight_submissions = 8;
            caps.max_timeout_ms = 10_000;
            caps.ddr_data_width_bits = 512;
            caps.dax_alignment_bytes = 4096;
            caps.dax_bytes = 32 * 1024 * 1024 * 1024;
            caps.per_lane_counter_mask = 0x0f;
            caps.accelerator_clock_hz = 400_000_000;
            Ok(())
        }
        fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            buffer.handle = self.next_handle;
            self.next_handle += 1;
            buffer.generation = 1;
            self.buffers.insert(
                buffer.handle,
                (buffer.dpa_offset, buffer.length, buffer.format, 1),
            );
            Ok(())
        }
        fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            self.buffers.remove(&buffer.handle);
            Ok(())
        }
        fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
            commit.new_generation = commit.expected_generation + 1;
            self.buffers.get_mut(&commit.handle).unwrap().3 = commit.new_generation;
            Ok(())
        }
        fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
            self.pending = unsafe {
                std::slice::from_raw_parts(submit.tasks_ptr as *const TaskV3, submit.count as usize)
            }
            .to_vec();
            self.submitted
                .lock()
                .unwrap()
                .extend_from_slice(&self.pending);
            self.submission += 1;
            submit.submission_id = self.submission;
            Ok(())
        }
        fn wait(
            &mut self,
            wait: &mut WaitV3,
            completions: &mut [CompletionV3],
        ) -> Result<(), String> {
            let memory = &mut *self.dax.bytes.lock().unwrap();
            for (index, task) in self.pending.iter().enumerate() {
                let matrix_base = self.buffers[&task.matrix_handle].0 + task.matrix_offset;
                let input_base = self.buffers[&task.input_handle].0 + task.input_offset;
                let output_base = self.buffers[&task.output_handle].0 + task.output_offset;
                for batch_index in 0..task.batch as usize {
                    let input_row = input_base
                        .checked_add(task.input_stride_bytes * batch_index as u64)
                        .ok_or("fake input row overflow")?;
                    let output_row = output_base
                        .checked_add(task.output_stride_bytes * batch_index as u64)
                        .ok_or("fake output row overflow")?;
                    for row in 0..task.valid_out as usize {
                        let mut raw = 0_i64;
                        for column in 0..2048_usize {
                            let element = row * 2048 + column;
                            let byte = memory[matrix_base as usize + element / 4];
                            let code = (byte >> (2 * (element % 4))) & 3;
                            let weight = match code {
                                0 => 0_i64,
                                1 => 1,
                                3 => -1,
                                _ => return Err("reserved ternary code".into()),
                            };
                            let qoff = input_row as usize + column * 2;
                            let quant = i16::from_le_bytes([memory[qoff], memory[qoff + 1]]);
                            raw += weight * i64::from(quant);
                        }
                        let offset = output_row as usize + row * 8;
                        memory[offset..offset + 8].copy_from_slice(&raw.to_le_bytes());
                    }
                }
                let mut completion = CompletionV3::default();
                completion.request_id = task.request_id;
                completion.lane_used = index as u32 % MAX_LANES;
                completion.accelerator_cycles = 10;
                completion.matrix_bytes_read = TILE_PACKED_BYTES as u64 * u64::from(task.batch);
                completion.input_bytes_read = 4096 * u64::from(task.batch);
                completion.output_bytes_written = 16384 * u64::from(task.batch);
                completion.start_cycle = 100 + index as u64 * 20;
                completion.end_cycle = completion.start_cycle + 10;
                completions[index] = completion;
            }
            wait.completed = self.pending.len() as u32;
            self.pending.clear();
            Ok(())
        }
    }

    #[derive(Default)]
    struct CaptureOutput(Mutex<Vec<u8>>);
    impl OutputCopier for CaptureOutput {
        unsafe fn copy(&self, _pointer: usize, bytes: &[u8]) -> Result<(), String> {
            *self.0.lock().unwrap() = bytes.to_vec();
            Ok(())
        }
    }

    #[test]
    fn executor_stages_unique_v3_tasks_validates_raw_accumulates_and_copies_back() {
        let _capture_guard = capture_test_guard();
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 2,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 2,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [9; 32],
            signature,
        };
        let mut matrix = Vec::new();
        matrix.extend_from_slice(&block(3, false, 0x700));
        matrix.extend_from_slice(&block(5, true, 0x321));
        let mut activations = vec![0_u8; 2 * Q8_1_MMQ_BYTES];
        for ds4 in activations.chunks_exact_mut(Q8_1_MMQ_BYTES) {
            for sub in 0..4 {
                ds4[sub * 4..sub * 4 + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
                ds4[sub * 4 + 2..sub * 4 + 4].copy_from_slice(&0x3e00_u16.to_le_bytes());
                for (index, value) in ds4[16 + sub * 32..16 + (sub + 1) * 32]
                    .iter_mut()
                    .enumerate()
                {
                    *value = (index as i8 - 16) as u8;
                }
            }
        }
        let reference_fixture_path = Path::new(DEFAULT_LIBGGML);
        assert!(reference_fixture_path.is_file());
        let oracle_grid = validated_grid(Some(reference_fixture_path)).unwrap();
        let captured = capture_from_host(launch, &matrix, &activations, &oracle_grid).unwrap();
        let dax = MemoryDax::new(20 * 1024 * 1024);
        let mut session = V3Session::with_io(FakeV3::new(dax.clone())).unwrap();
        let copied = CaptureOutput::default();
        let result = execute_captured_with(&captured, &mut session, &dax, &copied, 0).unwrap();
        assert_eq!(result.physical.len(), 16);
        assert!(result
            .outputs
            .iter()
            .all(|value| value.is_finite() && *value != 0.0));
        // Generated independently by mmfreelm.ops.iq1s_reference using this
        // exact packed IQ1_S/DS4 fixture and the installed libggml above.
        let python_libggml_references = [-215.625_f32, -237.625_f32];
        for (actual, reference) in result.outputs.iter().zip(python_libggml_references) {
            assert!(
                (actual - reference).abs() <= 1e-4 + 1e-4 * reference.abs(),
                "oracle-backed fixture mismatch: actual={actual} reference={reference}"
            );
        }
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xc357a000, 0xc36da000]
        );
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| item.completed.task().request_id)
                .collect::<Vec<_>>(),
            (1..=16).collect::<Vec<_>>()
        );
        assert!(result
            .physical
            .iter()
            .all(|item| item.completed.submission_id() == 1));
        assert!(result
            .physical
            .iter()
            .all(|item| item.batch_first == 0 && item.batch_count == 1));
        assert_eq!(result.scheduler.descriptor_count(), 16);
        assert_eq!(result.scheduler.logical_items(), 16);
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| item.completed.task().matrix_offset)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1
        );
        assert!(result
            .physical
            .iter()
            .all(|item| item.completed.task().matrix_offset == 0));
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| item.completed.task().matrix_handle)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| item.completed.task().input_offset)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| item.completed.task().output_offset)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        assert_eq!(copied.0.lock().unwrap().len(), 8);
        let writes = dax.writes.lock().unwrap();
        assert_eq!(writes.len(), 16 * 3);
        assert_eq!(
            writes
                .iter()
                .filter(|(_, bytes)| *bytes == TILE_PACKED_BYTES)
                .map(|(offset, _)| *offset)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        let fixture = execution_fixture_json(&captured, &result, reference_fixture_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&fixture).unwrap();
        assert_eq!(json["iq1s_hex"].as_str().unwrap().len(), 200);
        assert_eq!(json["q8_1_mmq_hex"].as_str().unwrap().len(), 576);
        assert_eq!(json["request_ids"].as_array().unwrap().len(), 16);
        assert_eq!(json["outputs_f32_bits"].as_array().unwrap().len(), 2);
        assert_eq!(
            Path::new(json["libggml_path"].as_str().unwrap()),
            reference_fixture_path
        );
    }

    #[test]
    fn executor_slices_four_logical_rows_by_live_v3_batch_cap_and_preserves_output_order() {
        // Global environment lock always precedes the IQ1_S capture/cache lock. No other IQ1_S
        // test acquires these in the opposite order, so parallel test execution cannot deadlock.
        let _env_lock = crate::r#impl::test_env::lock();
        let _capture_guard = capture_test_guard();
        let _batch_limit_guard = RemovedEnvGuard::new("HETGPU_FPGA_BATCH_LIMIT");

        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 2,
            stride01: 1,
            ne10: 256,
            ne11: 4,
            stride11: 4,
            ne0: 2,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x41000,
            activation_ptr: 0x42000,
            output_ptr: 0x43000,
            allocation_generation: 1,
            content_hash: [0x4b; 32],
            signature,
        };
        let mut matrix = Vec::new();
        matrix.extend_from_slice(&block(3, false, 0x700));
        matrix.extend_from_slice(&block(5, true, 0x321));
        let mut activations = vec![0_u8; 8 * Q8_1_MMQ_BYTES];
        let scales = [
            (0x3800_u16, 0x3e00_u16),
            (0x3c00_u16, 0x4200_u16),
            (0x3e00_u16, 0x4480_u16),
            (0x4000_u16, 0x4600_u16),
        ];
        for k_record in 0..2 {
            for (batch, &(d, s)) in scales.iter().enumerate() {
                let record = &mut activations[(k_record * 4 + batch) * Q8_1_MMQ_BYTES
                    ..(k_record * 4 + batch + 1) * Q8_1_MMQ_BYTES];
                for sub in 0..4 {
                    record[sub * 4..sub * 4 + 2].copy_from_slice(&d.to_le_bytes());
                    record[sub * 4 + 2..sub * 4 + 4].copy_from_slice(&s.to_le_bytes());
                    for (index, value) in record[16 + sub * 32..16 + (sub + 1) * 32]
                        .iter_mut()
                        .enumerate()
                    {
                        *value = (index as i8 - 16) as u8;
                    }
                }
            }
        }

        let oracle = validated_grid(Some(Path::new(DEFAULT_LIBGGML))).unwrap();
        let captured = capture_from_host(launch, &matrix, &activations, &oracle).unwrap();
        let dax = MemoryDax::new(24 * 1024 * 1024);
        let (io, submitted) = FakeV3::with_max_batch(dax.clone(), 2);
        let mut session = V3Session::with_io(io).unwrap();
        let copied = CaptureOutput::default();
        let result = execute_captured_with(&captured, &mut session, &dax, &copied, 0).unwrap();

        let component_count = captured.matrix.component_iter().unwrap().len();
        let submitted = submitted.lock().unwrap();
        assert_eq!(submitted.len(), component_count * 2);
        assert_eq!(result.physical.len(), component_count * 2);
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| (item.batch_first, item.batch_count))
                .collect::<Vec<_>>(),
            (0..component_count)
                .flat_map(|_| [(0, 2), (2, 2)])
                .collect::<Vec<_>>()
        );
        assert!(submitted
            .iter()
            .all(|task| task.batch == 2 && task.lane == LANE_ANY));
        assert_eq!(
            submitted
                .iter()
                .map(|task| task.request_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            submitted.len()
        );
        for component_tasks in submitted.chunks_exact(2) {
            let first = component_tasks[0];
            let second = component_tasks[1];
            assert_eq!(first.matrix_offset, 0);
            assert_eq!(second.matrix_offset, 0);
            assert_eq!(first.matrix_handle, second.matrix_handle);
            assert_eq!(
                second.input_offset,
                first.input_offset + 2 * first.input_stride_bytes
            );
            assert_eq!(
                second.output_offset,
                first.output_offset + 2 * first.output_stride_bytes
            );
            assert!(first.input_stride_bytes.is_multiple_of(4096));
            assert!(first.output_stride_bytes.is_multiple_of(4096));
        }
        assert_eq!(
            submitted
                .chunks_exact(2)
                .map(|tasks| tasks[0].matrix_handle)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            component_count
        );
        let mut input_ranges = submitted
            .iter()
            .map(|task| {
                (
                    task.input_offset,
                    task.input_offset + u64::from(task.batch) * task.input_stride_bytes,
                )
            })
            .collect::<Vec<_>>();
        input_ranges.sort_unstable();
        assert!(input_ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        let mut output_ranges = submitted
            .iter()
            .map(|task| {
                (
                    task.output_offset,
                    task.output_offset + u64::from(task.batch) * task.output_stride_bytes,
                )
            })
            .collect::<Vec<_>>();
        output_ranges.sort_unstable();
        assert!(output_ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                -215.625_f32,
                -237.625_f32,
                -431.25_f32,
                -475.25_f32,
                -646.875_f32,
                -712.875_f32,
                -862.5_f32,
                -950.5_f32,
            ]
            .map(f32::to_bits)
        );
        assert_eq!(copied.0.lock().unwrap().len(), 4 * 2 * 4);
        assert_eq!(result.raw_components.len(), component_count * 4);
        assert_eq!(
            result.scheduler.descriptor_count(),
            (component_count * 2) as u64
        );
        assert_eq!(
            result.scheduler.logical_items(),
            (component_count * 4) as u64
        );
        assert_eq!(result.scheduler.unique_submission_count(), 1);
        assert_ne!(result.scheduler.lane_mask(), 0);
        assert_eq!(
            result
                .scheduler
                .per_lane_completion_counts()
                .iter()
                .sum::<u64>(),
            (component_count * 2) as u64
        );
        assert!(session.io().buffers.is_empty());
    }

    fn parse_live_base_dpa(value: &str) -> Result<u64, String> {
        let value = value.trim();
        if let Some(hex) = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
        {
            u64::from_str_radix(hex, 16)
                .map_err(|_| format!("invalid HETGPU_CXL_TMATMUL_V3_BASE_DPA: {value}"))
        } else {
            value
                .parse::<u64>()
                .map_err(|_| format!("invalid HETGPU_CXL_TMATMUL_V3_BASE_DPA: {value}"))
        }
    }

    #[test]
    fn live_v3_host_captured_batch_four_fixture() {
        if std::env::var("HETGPU_RUN_LIVE_V3_BATCH").as_deref() != Ok("1") {
            eprintln!(
                "SKIP live_v3_host_captured_batch_four_fixture: set HETGPU_RUN_LIVE_V3_BATCH=1"
            );
            return;
        }

        let _env_lock = crate::r#impl::test_env::lock();
        let _capture_guard = capture_test_guard();
        let fixture_batch_limit = std::env::var("HETGPU_FPGA_BATCH_LIMIT")
            .expect("live fixture requires HETGPU_FPGA_BATCH_LIMIT=2");
        assert_eq!(
            fixture_batch_limit, "2",
            "live fixture must prove exactly two batch-2 slices"
        );
        let control_path = std::env::var("HETGPU_CXL_TMATMUL_DEVICE")
            .unwrap_or_else(|_| "/dev/cxl_tmatmul3b001".into());
        let dax_path =
            std::env::var("HETGPU_CXL_TMATMUL_DAX").unwrap_or_else(|_| "/dev/dax6.0".into());
        let base_dpa = std::env::var("HETGPU_CXL_TMATMUL_V3_BASE_DPA")
            .map(|value| parse_live_base_dpa(&value))
            .unwrap_or(Ok(0x0100_0000))
            .expect("valid live v3 base DPA");

        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 2,
            stride01: 1,
            ne10: 256,
            ne11: 4,
            stride11: 4,
            ne0: 2,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x41000,
            activation_ptr: 0x42000,
            output_ptr: 0x43000,
            allocation_generation: 1,
            content_hash: [0x4b; 32],
            signature,
        };
        let mut matrix = Vec::new();
        matrix.extend_from_slice(&block(3, false, 0x700));
        matrix.extend_from_slice(&block(5, true, 0x321));
        let mut activations = vec![0_u8; 8 * Q8_1_MMQ_BYTES];
        let scales = [
            (0x3800_u16, 0x3e00_u16),
            (0x3c00_u16, 0x4200_u16),
            (0x3e00_u16, 0x4480_u16),
            (0x4000_u16, 0x4600_u16),
        ];
        for k_record in 0..2 {
            for (batch, &(d, s)) in scales.iter().enumerate() {
                let record = &mut activations[(k_record * 4 + batch) * Q8_1_MMQ_BYTES
                    ..(k_record * 4 + batch + 1) * Q8_1_MMQ_BYTES];
                for sub in 0..4 {
                    record[sub * 4..sub * 4 + 2].copy_from_slice(&d.to_le_bytes());
                    record[sub * 4 + 2..sub * 4 + 4].copy_from_slice(&s.to_le_bytes());
                    for (index, value) in record[16 + sub * 32..16 + (sub + 1) * 32]
                        .iter_mut()
                        .enumerate()
                    {
                        *value = (index as i8 - 16) as u8;
                    }
                }
            }
        }

        let oracle = validated_grid(Some(Path::new(DEFAULT_LIBGGML)))
            .expect("load the validated local libggml IQ1_S grid");
        let captured = capture_from_host(launch, &matrix, &activations, &oracle)
            .expect("capture host IQ1_S batch fixture");
        let (mut session, bound_dax) =
            V3Session::open(Path::new(&control_path), Path::new(&dax_path))
                .expect("bind live devdax to the v3 session");
        let caps = session.caps();
        let dax = FileDaxAccess::map(&bound_dax).expect("mmap bound live DAX device");
        assert_eq!(caps.version, 3, "live fixture requires QUERY_CAPS_V3");
        assert!(
            caps.num_instances >= MAX_LANES,
            "live fixture requires at least {MAX_LANES} advertised lanes"
        );
        assert!(
            caps.max_batch >= 2,
            "live fixture requires caps.max_batch >= batch-2 slices"
        );

        let copied = CaptureOutput::default();
        let result = execute_captured_with(&captured, &mut session, &dax, &copied, base_dpa)
            .expect("execute live host-captured IQ1_S batch fixture");
        let component_count = captured
            .matrix
            .component_iter()
            .expect("iterate live fixture components")
            .len();
        let expected_descriptor_count = component_count
            .checked_mul(2)
            .expect("live descriptor count overflow");
        let expected_slices = (0..component_count)
            .flat_map(|_| [(0_u32, 2_u32), (2_u32, 2_u32)])
            .collect::<Vec<_>>();
        assert_eq!(result.physical.len(), expected_descriptor_count);
        assert_eq!(
            result
                .physical
                .iter()
                .map(|item| (item.batch_first, item.batch_count))
                .collect::<Vec<_>>(),
            expected_slices,
            "live descriptors must be component-major with two ordered batch-2 slices"
        );
        for component_slices in result.physical.chunks_exact(2) {
            assert_eq!(component_slices[0].batch_first, 0);
            assert_eq!(component_slices[1].batch_first, 2);
            assert!(component_slices.iter().all(|item| item.batch_count == 2
                && item.completed.task().batch == 2
                && item.completed.task().lane == LANE_ANY));
        }
        let expected_bits = [
            -215.625_f32,
            -237.625_f32,
            -431.25_f32,
            -475.25_f32,
            -646.875_f32,
            -712.875_f32,
            -862.5_f32,
            -950.5_f32,
        ]
        .map(f32::to_bits);
        let actual_bits = result
            .outputs
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual_bits, expected_bits, "live output must be bit-exact");
        assert_eq!(
            copied.0.lock().unwrap().as_slice(),
            output_bytes(&captured, &result.outputs)
                .expect("serialize validated output")
                .as_slice()
        );

        let report = &result.scheduler;
        assert_eq!(report.descriptor_count(), expected_descriptor_count as u64);
        let evidence = serde_json::json!({
            "event": "hetgpu_v3_live_batch_fixture_completed",
            "validated": true,
            "control_device": control_path,
            "dax_device": dax_path,
            "base_dpa": base_dpa,
            "caps": {
                "version": caps.version,
                "num_instances": caps.num_instances,
                "max_lanes": MAX_LANES,
                "max_batch": caps.max_batch,
                "max_descriptors": caps.max_descriptors,
            },
            "logical_batch": 4,
            "configured_batch_limit": 2,
            "component_count": component_count,
            "expected_descriptor_count": expected_descriptor_count,
            "ordered_component_major_slices": result.physical.iter().map(|item| {
                serde_json::json!({
                    "batch_first": item.batch_first,
                    "batch_count": item.batch_count,
                })
            }).collect::<Vec<_>>(),
            "output_f32_bits": actual_bits,
            "descriptor_count": report.descriptor_count(),
            "logical_items": report.logical_items(),
            "submission_count": report.unique_submission_count(),
            "lane_mask": report.lane_mask(),
            "per_lane_completion_counts": report.per_lane_completion_counts(),
            "accelerator_cycles": report.total_accelerator_cycles(),
            "matrix_bytes_read": report.total_matrix_bytes_read(),
            "input_bytes_read": report.total_input_bytes_read(),
            "output_bytes_written": report.total_output_bytes_written(),
        });
        println!(
            "{}",
            serde_json::to_string(&evidence).expect("serialize live completion evidence")
        );
    }
}
