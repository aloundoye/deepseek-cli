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
    /// GPU/accelerator memory in MB (e.g. Metal unified memory on macOS, VRAM on CUDA).
    /// `None` if no GPU detected or memory query failed.
    pub gpu_memory_mb: Option<u64>,
    /// Number of performance cores (Apple Silicon P-cores, or `None` on other platforms).
    pub performance_cores: Option<u32>,
    /// Number of efficiency cores (Apple Silicon E-cores, or `None` on other platforms).
    pub efficiency_cores: Option<u32>,
    /// Recommended thread count for inference. Uses P-cores when available,
    /// otherwise falls back to `available_parallelism() / 2`.
    pub recommended_threads: u32,
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
            let gpu_memory_mb = detect_gpu_memory(device, total_ram_mb);
            let (performance_cores, efficiency_cores) = detect_core_types();
            let recommended_threads = performance_cores.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| (n.get() as u32) / 2)
                    .unwrap_or(2)
                    .max(1)
            });
            HardwareInfo {
                device,
                total_ram_mb,
                available_for_models_mb,
                gpu_memory_mb,
                performance_cores,
                efficiency_cores,
                recommended_threads,
            }
        })
        .clone()
}

/// Return current available (free) system memory in MB.
///
/// Unlike `detect_hardware()` (which is cached once per process), this queries
/// live memory pressure every time it is called.
///
/// - macOS: `sysctl vm.page_free_count` * `hw.pagesize`
/// - Linux: `MemAvailable` from `/proc/meminfo`
/// - Fallback: re-uses `detect_hardware().available_for_models_mb`
pub fn available_memory_mb() -> u64 {
    available_memory_mb_impl().unwrap_or_else(|| detect_hardware().available_for_models_mb)
}

#[cfg(target_os = "macos")]
fn available_memory_mb_impl() -> Option<u64> {
    // Read free page count and page size from sysctl
    let free_output = Command::new("sysctl")
        .args(["-n", "vm.page_free_count"])
        .output()
        .ok()?;
    if !free_output.status.success() {
        return None;
    }
    let free_pages: u64 = String::from_utf8_lossy(&free_output.stdout)
        .trim()
        .parse()
        .ok()?;

    let page_output = Command::new("sysctl")
        .args(["-n", "hw.pagesize"])
        .output()
        .ok()?;
    if !page_output.status.success() {
        return None;
    }
    let page_size: u64 = String::from_utf8_lossy(&page_output.stdout)
        .trim()
        .parse()
        .ok()?;

    Some(free_pages * page_size / (1024 * 1024))
}

#[cfg(target_os = "linux")]
fn available_memory_mb_impl() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn available_memory_mb_impl() -> Option<u64> {
    None
}

/// Detect GPU memory in MB.
///
/// - macOS Metal (Apple Silicon): uses unified memory, reports ~75% of total RAM
///   as available for Metal (the OS reserves the rest). Queries `sysctl iogpu.wired_limit_mb`
///   if available, otherwise estimates.
/// - Linux CUDA: queries `nvidia-smi` for total GPU memory.
/// - CPU-only: returns None.
fn detect_gpu_memory(device: DetectedDevice, total_ram_mb: u64) -> Option<u64> {
    match device {
        DetectedDevice::Metal => {
            // Try IOKit query via system_profiler
            if let Some(mb) = macos_metal_memory_mb() {
                return Some(mb);
            }
            // Fallback: Apple Silicon uses unified memory, ~75% available for GPU
            Some(total_ram_mb * 3 / 4)
        }
        #[cfg(target_os = "linux")]
        DetectedDevice::Cuda => nvidia_gpu_memory_mb(),
        #[cfg(not(target_os = "linux"))]
        DetectedDevice::Cuda => None,
        DetectedDevice::Cpu => None,
    }
}

/// Query Metal GPU memory on macOS via `sysctl iogpu.wired_limit_mb`.
#[cfg(target_os = "macos")]
fn macos_metal_memory_mb() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "iogpu.wired_limit_mb"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    let mb: u64 = s.trim().parse().ok()?;
    // Treat 0 as unavailable — fall back to estimation
    if mb > 0 { Some(mb) } else { None }
}

#[cfg(not(target_os = "macos"))]
fn macos_metal_memory_mb() -> Option<u64> {
    None
}

/// Query NVIDIA GPU total memory via `nvidia-smi`.
#[cfg(target_os = "linux")]
fn nvidia_gpu_memory_mb() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    // Take the first GPU's memory
    s.lines().next()?.trim().parse().ok()
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

/// Detect P-core and E-core counts (Apple Silicon via sysctl).
///
/// Returns `(performance_cores, efficiency_cores)`.
/// On non-macOS-aarch64 platforms, both are `None`.
fn detect_core_types() -> (Option<u32>, Option<u32>) {
    let p = sysctl_u32("hw.perflevel0.physicalcpu");
    let e = sysctl_u32("hw.perflevel1.physicalcpu");
    (p, e)
}

/// Query a sysctl value as u32. Returns `None` on non-macOS or on any error.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_u32(key: &str) -> Option<u32> {
    let output = Command::new("sysctl").args(["-n", key]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn sysctl_u32(_key: &str) -> Option<u32> {
    None
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

    #[test]
    fn hardware_info_has_recommended_threads() {
        let hw = detect_hardware();
        assert!(
            hw.recommended_threads >= 1,
            "recommended_threads must be at least 1, got {}",
            hw.recommended_threads
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_detects_core_types() {
        let hw = detect_hardware();
        assert!(
            hw.performance_cores.is_some(),
            "Apple Silicon should detect P-cores"
        );
        let p = hw.performance_cores.unwrap();
        assert!(p >= 2, "expected at least 2 P-cores, got {p}");
        // E-cores may or may not exist (some M-series chips have them)
    }

    #[test]
    fn available_memory_returns_nonzero() {
        let mb = available_memory_mb();
        assert!(mb > 0, "available_memory_mb should return > 0, got {mb}");
    }

    #[test]
    fn available_memory_not_cached() {
        // Calling twice should succeed (not panic). Values may differ slightly
        // between calls because the function queries live memory pressure.
        let a = available_memory_mb();
        let b = available_memory_mb();
        // Both should be reasonable (> 0 and < total)
        let total = detect_hardware().total_ram_mb;
        assert!(a > 0 && a <= total, "first call out of range: {a}");
        assert!(b > 0 && b <= total, "second call out of range: {b}");
    }

    #[test]
    fn gpu_memory_detected_on_gpu_platforms() {
        let hw = detect_hardware();
        match hw.device {
            DetectedDevice::Metal | DetectedDevice::Cuda => {
                assert!(
                    hw.gpu_memory_mb.is_some(),
                    "GPU memory should be detected for {:?}",
                    hw.device
                );
                let gpu_mb = hw.gpu_memory_mb.unwrap();
                assert!(gpu_mb > 0, "GPU memory should be > 0");
            }
            DetectedDevice::Cpu => {
                // GPU memory is None for CPU-only systems
                assert!(hw.gpu_memory_mb.is_none());
            }
        }
    }
}
