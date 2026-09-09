#![allow(
    clippy::derivable_impls,
    clippy::doc_lazy_continuation,
    clippy::empty_line_after_doc_comments,
    clippy::identity_op,
    clippy::implicit_saturating_sub,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::unnecessary_unwrap
)]

pub mod arch;
pub mod backward;
pub mod config;
pub mod gpu;
pub mod model;
pub mod plan;
pub mod spec;

pub use arch::{Arch, ArchTrait, BaselineArch};
pub use backward::{ArtifactRegularizationConfig, ArtifactRegularizationReport, GradBuffers};
pub use config::{ModelConfig, TrainConfig};
pub use model::{ForwardBuffer, GptModel};
pub use plan::ExecutionPlan;
pub use spec::{
    AsymLogitSpec, AttentionBackend, AttnOutGateSpec, BackwardChainProfile, CompressionMode,
    CudaGraphProfile, DistributedOptimizerBackend, EvalAdaptationBackend, EvalSpec,
    ModelComputePrecision, ModelSpec, NcclOverlapMode, NgramTiltSpec, OutputCeBackend,
    QkvNormResidReducerProfile, QuantScheme, QuantSpec, RecordProfile, RecurrentBackwardProfile,
    RunMode, RunSpec, RuntimeSpec, ShortDocScoreFirstEntry, TrainBackend, TrainSeqScheduleEntry,
    TrainSpec, TttMask, VariantFamily,
};
