#![recursion_limit = "256"]
#![allow(
    clippy::collapsible_if,
    clippy::empty_line_after_doc_comments,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

pub mod compress;
pub mod export;
pub mod int6;
pub mod layout;
pub mod pack;
pub mod prune;
pub mod scheme;
pub mod serialize;

pub use layout::{
    CompiledQuantGroupManifest, CompiledQuantLayout, CompiledQuantLayoutManifest,
    FRONTIER_1855_MIXED_INT5_INT6_LAYOUT, FRONTIER_2135_MIXED_INT5_INT4_LAYOUT, QuantArchProfile,
    compile_quant_layout_manifest,
};
pub use pack::CompiledQuantKernelSet;
pub use pg_quant_macros::quant_layout;
