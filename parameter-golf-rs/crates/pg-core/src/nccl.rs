/// NCCL multi-GPU communication wrapper.
///
/// This module intentionally exposes correctness-first collectives only. The
/// train loop must still gate distributed record runs until GPU backward and
/// distributed optimizer code consume these methods end to end.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
#[cfg(feature = "cuda")]
use cudarc::nccl::{Comm, ReduceOp};

#[cfg(feature = "cuda")]
use crate::{GpuTensor, PgError, PgResult};

#[cfg(feature = "cuda")]
struct RawF32Buffer {
    ptr: u64,
    len: usize,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl DeviceSlice<f32> for RawF32Buffer {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[cfg(feature = "cuda")]
impl DevicePtr<f32> for RawF32Buffer {
    fn device_ptr<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        // These wrappers are only used on the communicator's owning stream and
        // for tensors already synchronized through the same training stream.
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
impl DevicePtrMut<f32> for RawF32Buffer {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
pub struct NcclComm {
    pub rank: usize,
    pub world_size: usize,
    comm: Option<Comm>,
}

#[cfg(not(feature = "cuda"))]
pub struct NcclComm {
    pub rank: usize,
    pub world_size: usize,
}

#[cfg(feature = "cuda")]
impl NcclComm {
    /// Metadata-only constructor retained for CPU/data-shard preflight code.
    /// Collective methods return an error unless the communicator came from
    /// `from_local_devices`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            comm: None,
        }
    }

    /// Build one NCCL communicator per local CUDA stream/device.
    pub fn from_local_devices(streams: Vec<Arc<CudaStream>>) -> PgResult<Vec<Self>> {
        let world_size = streams.len();
        let comms = Comm::from_devices(streams)
            .map_err(|e| PgError::Nccl(format!("Comm::from_devices failed: {e:?}")))?;
        Ok(comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| Self {
                rank,
                world_size,
                comm: Some(comm),
            })
            .collect())
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    pub fn all_reduce_sum_f32(
        &self,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
    ) -> PgResult<()> {
        self.comm()?
            .all_reduce::<_, _, f32>(send, recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_reduce_sum_f32 failed: {e:?}")))
    }

    pub fn reduce_scatter_sum_f32(
        &self,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
    ) -> PgResult<()> {
        self.comm()?
            .reduce_scatter::<_, _, f32>(send, recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_scatter_sum_f32 failed: {e:?}")))
    }

    pub fn all_gather_f32(&self, send: &CudaSlice<f32>, recv: &mut CudaSlice<f32>) -> PgResult<()> {
        self.comm()?
            .all_gather::<_, _, f32>(send, recv)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_gather_f32 failed: {e:?}")))
    }

    pub fn all_reduce_sum_tensor_f32_in_place(&self, tensor: &mut GpuTensor) -> PgResult<()> {
        if tensor.dtype() != crate::DType::F32 {
            return Err(PgError::Nccl(format!(
                "all_reduce_sum_tensor_f32_in_place requires F32 tensor, got {:?}",
                tensor.dtype()
            )));
        }
        let stream = self.comm()?.stream();
        let ptr = tensor.cu_ptr(&stream)?;
        let mut raw = RawF32Buffer {
            ptr,
            len: tensor.numel(),
            stream,
        };
        self.comm()?
            .all_reduce_in_place::<_, f32>(&mut raw, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_reduce_sum_tensor_f32_in_place failed: {e:?}")))
    }

    fn comm(&self) -> PgResult<&Comm> {
        self.comm.as_ref().ok_or_else(|| {
            PgError::Nccl(
                "NCCL communicator is metadata-only; construct with NcclComm::from_local_devices"
                    .into(),
            )
        })
    }
}

#[cfg(not(feature = "cuda"))]
impl NcclComm {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}
