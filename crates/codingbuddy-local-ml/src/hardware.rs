//! Hardware detection for automatic model selection.
//!
//! Pure `std` — no Candle deps, works without the `local-ml` feature.
//! Detects GPU availability and total system RAM to recommend the best model.

use std::process::Command;
use std::sync::OnceLock;

/// Detected compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedDevice {
    Metal,
    Cuda,
    Cpu,
}

impl std::fmt::Display for DetectedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectedDevice::Metal => write!(f, "metal"),
            DetectedDevice::Cuda => write!(f, "cuda"),
            DetectedDevice::Cpu => write!(f, "cpu"),
        }
    }
}

/// Hardware information for model selection.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Detected compute device.
    pub device: DetectedDevice,
    /// Total system RAM in MB.
    pub total_ram_mb: u64,
    /// RAM available for model loading (total minus 4 GB OS reserve).
    pub available_for_models_mb: u64,
}

const OS_RESERVE_MB: u64 = 4096;

/// Detect hardware capabilities (GPU type and available RAM).
///
/// Results are cached — subprocess calls happen at most once per process.
///
/// - macOS aarch64: Metal GPU, RAM via `sysctl hw.memsize`
/// - Linux: checks for NVIDIA GPU via `nvidia-smi`, RAM via `/proc/meminfo`
/// - Fallback: CPU with 8 GB assumed
pub fn detect_hardware() -> HardwareInfo {
    static CACHE: OnceLock<HardwareInfo> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let (device, total_ram_mb) = detect_platform();
            let available_for_models_mb = total_ram_mb.saturating_sub(OS_RESERVE_MB);
            HardwareInfo {
                device,
                total_ram_mb,
                available_for_models_mb,
            }
        })
        .clone()
}

#[cfg(target_os = "macos")]
fn detect_platform() -> (DetectedDevice, u64) {
    let device = if cfg!(target_arch = "aarch64") {
        DetectedDevice::Metal
    } else {
        DetectedDevice::Cpu
    };
    let ram = macos_total_ram_mb().unwrap_or(8192);
    (device, ram)
}

#[cfg(target_os = "linux")]
fn detect_platform() -> (DetectedDevice, u64) {
    let device = if has_nvidia_gpu() {
        DetectedDevice::Cuda
    } else {
        DetectedDevice::Cpu
    };
    let ram = linux_total_ram_mb().unwrap_or(8192);
    (device, ram)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_platform() -> (DetectedDevice, u64) {
    (DetectedDevice::Cpu, 8192)
}

/// Read total RAM on macOS via `sysctl -n hw.memsize` (returns bytes).
#[cfg(target_os = "macos")]
fn macos_total_ram_mb() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    let bytes: u64 = s.trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

/// Check for NVIDIA GPU by running `nvidia-smi`.
#[cfg(target_os = "linux")]
fn has_nvidia_gpu() -> bool {
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Read total RAM on Linux from `/proc/meminfo`.
#[cfg(target_os = "linux")]
fn linux_total_ram_mb() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // Format: "MemTotal:       16384000 kB"
            let kb_str = rest
                .trim()
                .strip_suffix("kB")
                .or_else(|| rest.trim().strip_suffix("KB"))?;
            let kb: u64 = kb_str.trim().parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hardware_returns_valid_info() {
        let hw = detect_hardware();
        assert!(hw.total_ram_mb > 0, "should detect some RAM");
        assert!(
            hw.available_for_models_mb <= hw.total_ram_mb,
            "available must be <= total"
        );
    }

    #[test]
    fn os_reserve_subtracted() {
        let hw = detect_hardware();
        if hw.total_ram_mb > OS_RESERVE_MB {
            assert_eq!(hw.available_for_models_mb, hw.total_ram_mb - OS_RESERVE_MB);
        } else {
            assert_eq!(hw.available_for_models_mb, 0);
        }
    }

    #[test]
    fn detected_device_display() {
        assert_eq!(DetectedDevice::Metal.to_string(), "metal");
        assert_eq!(DetectedDevice::Cuda.to_string(), "cuda");
        assert_eq!(DetectedDevice::Cpu.to_string(), "cpu");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_detects_correct_device() {
        let hw = detect_hardware();
        if cfg!(target_arch = "aarch64") {
            assert_eq!(hw.device, DetectedDevice::Metal);
        }
    }
}
