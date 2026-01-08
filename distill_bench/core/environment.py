"""
Hardware and Software Environment Metadata Collection.

Collects detailed information about the execution environment for reproducibility:
- GPU specifications (via NVML)
- CPU specifications (via psutil)
- System memory
- Software versions (CUDA, PyTorch, Transformers)
- Git repository state
- SLURM job information (if available)
"""

import os
import platform
import subprocess
import json
from typing import Dict, Any, List
from pathlib import Path

import pynvml
import psutil
import torch
import transformers


def collect_gpu_info() -> List[Dict[str, Any]]:
    """Collect GPU information via NVML."""
    gpus = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Versions
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")

            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

            # Compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

            # Power limit
            try:
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                power_limit_watts = power_limit_mw / 1000.0
            except pynvml.NVMLError:
                power_limit_watts = 0.0

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_mb": mem_info.total / (1024**2),
                    "memory_free_mb": mem_info.free / (1024**2),
                    "driver_version": driver_version,
                    "cuda_version": cuda_version_str,
                    "compute_capability": f"{major}.{minor}",
                    "power_limit_watts": power_limit_watts,
                }
            )

        pynvml.nvmlShutdown()

    except pynvml.NVMLError as e:
        print(f"Warning: Failed to collect GPU info: {e}")

    return gpus


def collect_cpu_info() -> Dict[str, Any]:
    """Collect CPU information."""
    # Try to get CPU brand from /proc/cpuinfo on Linux
    cpu_brand = "unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_brand = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, IndexError):
        cpu_brand = platform.processor() or "unknown"

    physical_cores = psutil.cpu_count(logical=False) or 0
    logical_cores = psutil.cpu_count(logical=True) or 0

    # CPU frequency
    try:
        freq = psutil.cpu_freq()
        max_freq = freq.max if freq else 0.0
        current_freq = freq.current if freq else 0.0
    except (AttributeError, OSError):
        max_freq = 0.0
        current_freq = 0.0

    return {
        "brand": cpu_brand,
        "architecture": platform.machine(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "max_frequency_mhz": max_freq,
        "current_frequency_mhz": current_freq,
    }


def collect_memory_info() -> Dict[str, Any]:
    """Collect system memory information."""
    mem = psutil.virtual_memory()

    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent_used": mem.percent,
    }


def collect_software_info() -> Dict[str, Any]:
    """Collect software version information."""
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
    cudnn_version = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None
    transformers_version = transformers.__version__

    return {
        "python_version": platform.python_version(),
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "transformers_version": transformers_version,
        "platform": platform.system(),
        "os_version": platform.release(),
    }


def collect_git_info() -> Dict[str, Any]:
    """Collect git repository information."""
    git_info = {
        "commit_hash": None,
        "branch": None,
        "is_dirty": False,
        "remote_url": None,
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["is_dirty"] = bool(result.stdout.strip())

        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass  # Git not available or not in a git repo

    return git_info


def collect_slurm_info() -> Dict[str, Any]:
    """Collect SLURM job information if running in SLURM."""
    slurm_info = {}

    # Common SLURM environment variables
    slurm_vars = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "node_name": "SLURM_NODELIST",
        "partition": "SLURM_JOB_PARTITION",
        "num_nodes": "SLURM_JOB_NUM_NODES",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
        "gpus": "SLURM_GPUS_ON_NODE",
        "mem_per_node": "SLURM_MEM_PER_NODE",
    }

    for key, env_var in slurm_vars.items():
        value = os.environ.get(env_var)
        if value is not None:
            slurm_info[key] = value

    return slurm_info


def collect_environment() -> Dict[str, Any]:
    """Collect all environment metadata."""
    return {
        "hostname": platform.node(),
        "gpus": collect_gpu_info(),
        "cpu": collect_cpu_info(),
        "memory": collect_memory_info(),
        "software": collect_software_info(),
        "git_info": collect_git_info(),
        "slurm_info": collect_slurm_info(),
    }


def print_environment(env: Dict[str, Any]):
    """Print formatted environment summary."""
    print("=" * 70)
    print("ENVIRONMENT METADATA")
    print("=" * 70)

    print(f"\nHostname: {env['hostname']}")

    # GPUs
    gpus = env.get("gpus", [])
    print(f"\n--- GPUs ({len(gpus)}) ---")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_total_mb']:.0f} MB")
        print(f"    Driver: {gpu['driver_version']}, CUDA: {gpu['cuda_version']}")
        print(f"    Compute: {gpu['compute_capability']}, Power Limit: {gpu['power_limit_watts']:.0f}W")

    # CPU
    cpu = env.get("cpu", {})
    print(f"\n--- CPU ---")
    print(f"  Model: {cpu.get('brand', 'unknown')}")
    print(f"  Cores: {cpu.get('physical_cores', 0)} physical, {cpu.get('logical_cores', 0)} logical")
    print(f"  Frequency: {cpu.get('current_frequency_mhz', 0):.0f} MHz (max {cpu.get('max_frequency_mhz', 0):.0f} MHz)")

    # Memory
    mem = env.get("memory", {})
    print(f"\n--- Memory ---")
    print(f"  Total: {mem.get('total_gb', 0):.1f} GB")
    print(f"  Available: {mem.get('available_gb', 0):.1f} GB")
    print(f"  Used: {mem.get('used_gb', 0):.1f} GB ({mem.get('percent_used', 0):.1f}%)")

    # Software
    sw = env.get("software", {})
    print(f"\n--- Software ---")
    print(f"  Python: {sw.get('python_version', 'unknown')}")
    print(f"  PyTorch: {sw.get('pytorch_version', 'unknown')}")
    print(f"  CUDA: {sw.get('cuda_version', 'N/A')}")
    print(f"  Transformers: {sw.get('transformers_version', 'unknown')}")
    print(f"  Platform: {sw.get('platform', 'unknown')}")

    # Git
    git = env.get("git_info", {})
    if git.get("commit_hash"):
        print(f"\n--- Git ---")
        print(f"  Commit: {git['commit_hash']}")
        print(f"  Branch: {git.get('branch', 'unknown')}")
        print(f"  Dirty: {git.get('is_dirty', False)}")

    # SLURM
    slurm = env.get("slurm_info", {})
    if slurm.get("job_id"):
        print(f"\n--- SLURM ---")
        print(f"  Job ID: {slurm['job_id']}")
        print(f"  Node: {slurm.get('node_name', 'unknown')}")
        print(f"  Partition: {slurm.get('partition', 'unknown')}")

    print("=" * 70)


def save_environment(output_dir: str, filename: str = "environment.json") -> Path:
    """Collect and save environment metadata to JSON file."""
    env = collect_environment()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(env, f, indent=2)

    print(f"Environment metadata saved to: {filepath}")
    return filepath


# Backward compatibility aliases
get_environment_metadata = collect_environment
save_environment_metadata = save_environment


# Example usage
if __name__ == "__main__":
    env = collect_environment()
    print_environment(env)

    output_path = Path("./environment_metadata.json")
    with open(output_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"\nSaved to: {output_path}")
