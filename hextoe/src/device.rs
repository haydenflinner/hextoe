//! Default compute device for inference and training.
//!
//! On macOS, when Candle is built with the `metal` feature, we prefer [`Device::new_metal`].
//! Candle does not expose a wgpu backend; GPU on Mac is Metal only.

use candle_core::{Device, Result};

/// Prefer Apple GPU (Metal) when this build includes Candle’s `metal` feature; otherwise CPU.
///
/// If Metal initialization fails, prints a short message to stderr and returns CPU.
pub fn default_inference_device() -> Device {
    try_default_inference_device().unwrap_or_else(|_| Device::Cpu)
}

/// Same as [`default_inference_device`] but surfaces initialization errors (before CPU fallback).
pub fn try_default_inference_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        if candle_core::utils::metal_is_available() {
            match Device::new_metal(0) {
                Ok(d) => return Ok(d),
                Err(e) => {
                    eprintln!("candle: Metal device unavailable ({e}); using CPU.");
                }
            }
        }
    }
    Ok(Device::Cpu)
}
