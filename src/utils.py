import os, csv, time, glob, sys, atexit, threading
from datetime import datetime
import pdb
from tqdm import tqdm
import shutil
import torch
import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from trl import DataCollatorForCompletionOnlyLM
import random
import numpy as np
import subprocess, json, re
from hashlib import blake2s
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import tempfile
import queue
import wandb

# ---------------------- Helper functions ----------------------

def _to_scalar(x: Any):
    """Convert various types to Python scalar."""
    # Works for Python numbers, PyTorch tensors, numpy scalars
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().float().item()
            raise ValueError("Tensor must be scalar")
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.number):
            return float(x)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        pass
    # Last resort
    return float(x)

# ---------------------- Training and FSDP2-specific functions ----------------------
try:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import Shard, DTensor
    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False
    FSDPModule = None
    DTensor = None

def inspect_model(model):
    """Inspect FSDP2 model structure and parameters."""
    if not FSDP2_AVAILABLE or not isinstance(model, FSDPModule):
        return
        
    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert isinstance(param, DTensor) 


def inspect_mixed_precision(model):
    """Inspect mixed precision settings for FSDP2 model."""
    if not FSDP2_AVAILABLE or not isinstance(model, FSDPModule):
        return
        
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

def check_batch_shape(train_dataloader):
    """Debug utility to check batch shapes."""
    batch = next(iter(train_dataloader))
    print("Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and len(value) > 0:
            # Check first few items and their types
            tensor_shapes = []
            other_types = []
            for i, v in enumerate(value[:3]):
                if isinstance(v, torch.Tensor):
                    tensor_shapes.append(v.shape)
                else:
                    other_types.append(type(v).__name__)
            
            if tensor_shapes:
                print(f"  {key}: tensor shapes {tensor_shapes}")
            if other_types:
                print(f"  {key}: also contains {other_types}")
            if not tensor_shapes and not other_types:
                print(f"  {key}: list of {len(value)} items, first type: {type(value[0]).__name__}")
        else:
            print(f"  {key}: {type(value).__name__}")

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Set forward prefetching for model layers."""
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Set backward prefetching for model layers."""
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


# ---------------------- Utility functions ----------------------

# Global event for SLURM signal handling
_exit_once = threading.Event()
_cleanup_once = threading.Event()

def slurm_term_handler(signum, frame, trainer, output_path=None, manifest=None):
    """Handle SLURM termination signals."""
    if _exit_once.is_set():
        return
    _exit_once.set()

    main_print(f"[signal {signum}] Preemption received; saving checkpoint...")
    trainer.save_checkpoint(None)
    main_print("Checkpoint saved.")

    # Update manifest status if available
    if manifest and is_main_process():
        try:
            manifest.set_status('FAILED')
            main_print("Manifest status updated to FAILED.")
        except Exception as e:
            main_print(f"Warning: Failed to update manifest: {e}")

    # Update status files if output_path available
    if output_path:
        status_running = os.path.join(output_path, "STATUS.RUNNING")
        status_done = os.path.join(output_path, "STATUS.DONE")
        status_failed = os.path.join(output_path, "STATUS.FAILED")
        
        # Clean up old status files
        for status_file in [status_failed, status_running, status_done]:
            if os.path.exists(status_file):
                os.remove(status_file)

        open(status_failed, 'a').close()
        main_print("Status files updated to FAILED.")

    os._exit(0)

def setup_exception_handling():
    """Set up custom exception handling and cleanup on exit."""
    # Store the original excepthook
    default_excepthook = sys.excepthook
    
    # Register cleanup function for normal exit (don't force error code)
    atexit.register(cleanup_and_exit, 0)
    
    # Set our custom exception handler
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Custom exception handler that ensures cleanup"""
        if exc_type is not None:
            main_print(f"Unhandled exception: {exc_value}")
            cleanup_and_exit()
            default_excepthook(exc_type, exc_value, exc_traceback)
            sys.exit(1)
    
    sys.excepthook = exception_handler

def cleanup_and_exit(exit_code=1):
    # Cleanup function to ensure proper resource cleanup on exit/error
    if _cleanup_once.is_set():
        return
    _cleanup_once.set()
    
    error_occurred = False
    try:
        if is_main_process():
            main_print("\n--> Cleaning up resources...")
            # Add any additional cleanup here
            if 'WANDB_RUN_ID' in os.environ:
                try:
                    if wandb.run is not None:
                        wandb.finish()
                except ImportError:
                    pass  # wandb not available
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        main_print(f"Error during cleanup: {e}")
        error_occurred = True
    finally:
        # Only exit if we're called during error handling
        if exit_code != 0 or error_occurred:
            os._exit(exit_code)

def exception_handler(exc_type, exc_value, exc_traceback):
    """Custom exception handler that ensures cleanup"""
    main_print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
    cleanup_and_exit()
    # Call original exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    

def init_wandb_run():
    """Initialize wandb run. Note: This function should be called after run_id is set."""
    try:
        import wandb
        # Get run_id from global scope or build it
        try:
            # Try to get from global
            run_id_local = globals().get('run_id')
            if run_id_local is None:
                # Build a temporary one
                run_id_local, _, _, wandb_id = build_run_identity()
            else:
                _, _, _, wandb_id = build_run_identity()
        except:
            # Fallback
            run_id_local = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb_id = run_id_local
            
        wandb_run = wandb.init(
            project="slm-ensembles",
            id=wandb_id,   
            name=run_id_local,
            config={
                "model_name": config.student_model_name,
                "teacher_model": config.teacher_model_name,
                "learning_rate": config.learning_rate,
                "batch_size": config.per_device_train_batch_size * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1),
                "max_length": 1024,
                "alpha": config.alpha,
                "seed": config.seed,
                "description": getattr(config, 'description', 'SLM ensemble training'),
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "total_rounds": config.total_rounds,
                "num_train_epochs": config.num_train_epochs,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_grad_norm": getattr(config, 'max_grad_norm', 1.0),
            },
            tags=["knowledge-distillation", "fsdp2", "ensemble"],
            resume="allow",
        )
        main_print(f"--> Initialized wandb run: {wandb_run.name}")
    except Exception as e:
        main_print(f"--> Warning: Failed to initialize wandb: {e}")
        main_print("--> Continuing without wandb logging")
        wandb_run = None
    return wandb_run

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main_print(*args, **kwargs):
    """Print only from main process in distributed training."""
    if is_main_process():
        print(*args, **kwargs)

def _git_short():
    """Retrieve the short Git commit hash of the current HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"

def _get_rank():
    """Get current process rank in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return _get_rank() == 0 if config.ddp else True


def format_time_elapsed(seconds):
    """Format elapsed time in minutes and seconds."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def get_round_path(output_path, round_num):
    """Get the path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")

# ---------------------- ID Tracking ----------------------
class AsyncLossLogger:
    """
    Append-only JSONL writer with batching and periodic atomic snapshots.
    Designed to keep training non-blocking.
    """
    def __init__(self, log_path: str, snapshot_path: str = None,
                 flush_interval_s: float = 1.0, snapshot_interval_s: float = 60.0,
                 max_queue: int = 100_000):
        self.log_path = log_path
        self.snapshot_path = snapshot_path or (os.path.splitext(log_path)[0] + ".snapshot.json")
        self.flush_interval_s = flush_interval_s
        self.snapshot_interval_s = snapshot_interval_s

        # Only main process creates the directory to avoid race conditions
        if is_main_process():
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        dist.barrier()
            
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._lock = threading.Lock()   # protects self.loss_dict during snapshot
        self.loss_dict: Dict[Any, Any] = {}  # live view for your code

        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

    def write(self, record: Dict[str, Any]):
        """Non-blocking enqueue; drops if queue is full to avoid stalling."""
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # If you want to block instead, use self._q.put(record)
            pass

    def update_and_write_many(self, ids, tr_step_loss, next_token_loss, kl_loss, valid_count):
        ids_list = list(ids)

        # Convert elementwise (each item can be a Python number, 0-d torch tensor, numpy scalar, etc.)
        tr_list = [_to_scalar(x) for x in tr_step_loss]
        nt_list = [_to_scalar(x) for x in next_token_loss]
        kl_list = [_to_scalar(x) for x in kl_loss]
        vc_list = [int(_to_scalar(x)) for x in valid_count]

        # --- Option A: strict length check (fail fast) ---
        if not (len(ids_list) == len(tr_list) == len(nt_list) == len(kl_list) == len(vc_list)):
            raise ValueError(
                f"Length mismatch: ids={len(ids_list)}, tr={len(tr_list)}, "
                f"nt={len(nt_list)}, kl={len(kl_list)}, vc={len(vc_list)}"
            )

        now = time.time()
        with self._lock:
            for id_, tr, nt, kl, vc in zip(ids_list, tr_list, nt_list, kl_list, vc_list):
                self.loss_dict[id_] = [tr, nt, kl, vc]
                self.write({"id": id_, "tr": tr, "nt": nt, "kl": kl, "vc": vc, "t": now})


    def close(self, timeout: float = 10.0):
        self._stop.set()
        self._writer.join(timeout=timeout)
        # Final snapshot on close
        self._save_snapshot()

    # ---- internals ----
    def _writer_loop(self):
        buf = []
        last_flush = time.time()
        last_snapshot = time.time()

        # Open once; let OS page cache buffer writes
        f = open(self.log_path, "a", buffering=1)  # line-buffered
        try:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    item = self._q.get(timeout=self.flush_interval_s)
                    buf.append(item)
                except queue.Empty:
                    pass

                now = time.time()
                if buf and (now - last_flush >= self.flush_interval_s or len(buf) >= 2048):
                    for rec in buf:
                        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    f.flush()  # do not fsync; keeps it fast
                    buf.clear()
                    last_flush = now

                if now - last_snapshot >= self.snapshot_interval_s:
                    # atomic snapshot of in-memory dict
                    self._save_snapshot()
                    last_snapshot = now
        finally:
            if buf:
                for rec in buf:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
            f.close()

    def _save_snapshot(self):
        try:
            with self._lock:
                tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(self.snapshot_path) or ".")
                json.dump(self.loss_dict, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())  # ensure durability of snapshot only
                tmp_path = tmp.name
                tmp.close()
            os.replace(tmp_path, self.snapshot_path)  # atomic
        except Exception as e:
            # Don't crash training if snapshot fails
            print(f"[AsyncLossLogger] snapshot failed: {e}")


def load_loss_jsonls(example_path: str) -> Tuple[Dict[Any, dict], List[dict]]:
    """
    Given a path to one JSONL file (e.g., ".../loss_log_0.jsonl"),
    load all sibling files whose names end with "_<number>.jsonl",
    parse all lines, and return:
      - by_id: dict mapping `id` -> latest row (by 't' if present, else last seen)
      - all_rows: list of every parsed row (in filename order, then line order)
    """
    p = Path(example_path)
    logs_dir = p.parent

    # Build a prefix like "loss_log" from "loss_log_0.jsonl"
    stem = p.name
    if "_" not in stem:
        raise ValueError(f"Expected an underscore before numeric suffix in filename: {p.name}")
    base_prefix = stem.rsplit("_", 1)[0]  # "loss_log"

    # Find files matching base_prefix_*.jsonl with numeric suffix
    suffix_rx = re.compile(rf"^{re.escape(base_prefix)}_(\d+)\.jsonl$")
    candidates = [f for f in logs_dir.glob(f"{base_prefix}_*.jsonl") if suffix_rx.match(f.name)]

    # Sort by numeric suffix so we have a deterministic read order
    def _suffix_num(f: Path) -> int:
        return int(suffix_rx.match(f.name).group(1))
    files = sorted(candidates, key=_suffix_num)

    by_id: Dict[Any, dict] = {}
    all_rows: List[dict] = []

    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip malformed lines but keep going
                    print(f"Skipping bad JSON in {f.name}:{lineno} -> {e}")
                    continue

                all_rows.append(row)
                k = row.get("id")
                if k is None:
                    # If there's no 'id', we just keep it in all_rows.
                    continue

                # Merge policy: keep the row with the latest 't' (timestamp) if available;
                # otherwise, last one wins.
                if k not in by_id:
                    by_id[k] = row
                else:
                    new_t = row.get("t")
                    old_t = by_id[k].get("t")
                    if new_t is None or old_t is None:
                        by_id[k] = row  # fall back to last-seen wins
                    elif new_t >= old_t:
                        by_id[k] = row

    return by_id, all_rows

def top_k_percent_ids_sorted(stats, k_percent):
    n = len(stats)
    top_n = max(1, math.ceil(n * k_percent / 100.0)) if n else 0
    return [k for k, _ in sorted(stats.items(), key=lambda kv: (-float(kv[1]["tr"]), kv[0]))][:top_n]

# ---------------------- Run ID and Directory Config Functions ----------------------

def _hp_fingerprint() -> str:
    """Generate a short hash fingerprint from config parameters."""
    keys = [
        "student_model_name", "teacher_model_name", "learning_rate", "alpha",
        "kl_temperature", "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_train_epochs", "total_rounds", "dataset_name", "dataset_type", "seed"
    ]
    blob = json.dumps({k: getattr(config, k) for k in keys}, sort_keys=True).encode()
    return blake2s(blob, digest_size=4).hexdigest()

def _abbr_model(name: str) -> str:
    """Abbreviate model name for concise identification."""
    base = name.split("/")[-1].lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", base)
    size = (m.group(1) + "b") if m else ""
    family = "qwen" if "qwen" in base else base.split("-")[0]
    return f"{family}{size.replace('.','p')}"

def build_run_identity():
    """Construct run identity components for logging and tracking."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    git = _git_short()
    fp = _hp_fingerprint()

    # Create readable hyperparameter string for directory naming
    hp_string = "-".join([
        _abbr_model(config.student_model_name),
        f"a{config.alpha}",
        f"t{config.kl_temperature}",
        f"lr{str(config.learning_rate).replace('.', 'p').replace('-', 'n')}",
    ])
    
    # Use readable hyperparameters instead of cryptic fingerprint
    run_id = f"{config.run_name}-{ts}-{git}-{hp_string}"
    
    slug = hp_string
    alias = os.environ.get("RUN_ALIAS")
    if alias:
        slug = f"{alias}-{slug}"

    wandb_name = f"{slug} | {run_id}"
    wandb_id = f"{fp}-{ts}"

    return run_id, slug, wandb_name, wandb_id

def get_directory(run_id):
    output_dir = config.base_output_dir
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir)
    return run_dir

# ---------------------- Manifest File Handling ----------------------

class ManifestManager:
    """
    Manages a manifest dictionary with automatic file synchronization.
    Provides human-readable INI-like format that's easy to cat and parse.
    """
    
    def __init__(self, manifest_path, auto_save=True):
        self.manifest_path = manifest_path
        self.auto_save = auto_save
        self.data = {}
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing manifest file if it exists"""
        if os.path.exists(self.manifest_path):
            self.load()
    
    def load(self):
        """Load manifest from file"""
        if not os.path.exists(self.manifest_path):
            return
        
        self.data = {}
        current_section = None
        
        with open(self.manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Section header [section_name]
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    self.data[current_section] = {}
                # Key-value pair
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert common types
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        value = float(value) if '.' in value else int(value)
                    elif value.lower() == 'none':
                        value = None
                    
                    if current_section:
                        self.data[current_section][key] = value
                    else:
                        self.data[key] = value
    
    def save(self):
        """Save manifest to file with atomic write"""
        lines = []
        
        # First write top-level keys
        for key, value in self.data.items():
            if not isinstance(value, dict):
                lines.append(f"{key}: {value}")
        
        # Add blank line if we had top-level keys
        if any(not isinstance(v, dict) for v in self.data.values()):
            lines.append("")
        
        # Then write sections
        for section_name, section_data in self.data.items():
            if isinstance(section_data, dict):
                lines.append(f"[{section_name}]")
                for key, value in section_data.items():
                    lines.append(f"{key}: {value}")
                lines.append("")  # blank line between sections
        
        # Atomic write
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, 'w') as f:
            f.write('\n'.join(lines))
        os.replace(tmp_path, self.manifest_path)
    
    def update(self, key_or_dict, value=None, section=None):
        """
        Update manifest data. Can be used in multiple ways:
        - update('key', 'value') - set top-level key
        - update('key', 'value', section='section_name') - set key in section
        - update({'key1': 'val1', 'key2': 'val2'}) - update multiple top-level keys
        - update({'key1': 'val1'}, section='section_name') - update multiple keys in section
        """
        if isinstance(key_or_dict, dict):
            # Bulk update
            if section:
                if section not in self.data:
                    self.data[section] = {}
                self.data[section].update(key_or_dict)
            else:
                self.data.update(key_or_dict)
        else:
            # Single key update
            if section:
                if section not in self.data:
                    self.data[section] = {}
                self.data[section][key_or_dict] = value
            else:
                self.data[key_or_dict] = value
        
        if self.auto_save:
            self.save()
    
    def get(self, key, section=None, default=None):
        """Get value from manifest"""
        if section:
            return self.data.get(section, {}).get(key, default)
        return self.data.get(key, default)
    
    def set_status(self, status):
        """Convenience method to update status"""
        self.update('status', status)
    
    def increment_round(self):
        """Convenience method to increment training round"""
        current_round = self.get('round', section='train', default=0)
        self.update('round', current_round + 1, section='train')
    
    def finalize(self, success=True, end_time=None, wall_time_sec=None):
        """Finalize the manifest with completion info"""
        if end_time is None:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.update({
            'end_time': end_time,
            'wall_time_sec': wall_time_sec,
        }, section='outcomes')
        
        self.set_status('DONE' if success else 'FAILED')


def create_manifest(output_path, start_time_str=None, wandb_run=None, wandb_id=None):
    """Create initial manifest file and return ManifestManager instance"""
    run_id = os.path.basename(output_path)
    manifest_path = os.path.join(output_path, "manifest.txt")
    
    # Status sentinels
    status_running = os.path.join(output_path, "STATUS.RUNNING")
    status_done = os.path.join(output_path, "STATUS.DONE")
    status_failed = os.path.join(output_path, "STATUS.FAILED")
    
    # Clean up old status files
    for status_file in [status_failed, status_running, status_done]:
        if os.path.exists(status_file):
            os.remove(status_file)
    
    # Resolve hardware info
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Timestamps
    start_time_str = start_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create manifest manager
    manifest = ManifestManager(manifest_path)
    
    # Initialize with all the data
    manifest.update({
        'run_id': run_id,
        'start_time': start_time_str,
        'run_dir': output_path,
        'git_commit': _git_short(),
        'status': 'RUNNING'
    })
    
    manifest.update({
        'id': wandb_id,
        'name': getattr(wandb_run, 'name', None) if wandb_run else None,
        'project': 'slm-ensembles',
    }, section='wandb')
    
    manifest.update({
        'ddp': getattr(config, 'ddp', False),
        'world_size': world_size,
        'devices': getattr(config, 'devices', 'unknown'),
    }, section='hardware')
    
    manifest.update({
        'dataset_name': config.dataset_name,
        'dataset_path': config.dataset_path,
        'synthetic_data': getattr(config, 'synthetic_data', False),
    }, section='data')
    
    manifest.update({
        'teacher': config.teacher_model_name,
        'student': config.student_model_name,
        'tokenizer': config.tokenizer_name,
    }, section='models')
    
    manifest.update({
        'alpha': config.alpha,
        'kl_temperature': config.kl_temperature,
        'learning_rate': config.learning_rate,
        'lr_scheduler_type': config.lr_scheduler_type,
        'weight_decay': getattr(config, 'weight_decay', 0.01),
        'warmup_steps': getattr(config, 'warmup_steps', 0),
        'num_train_epochs': config.num_train_epochs,
        'ensemble_models': config.total_rounds,
        'seed': config.seed,
        'round': 0,
    }, section='train')
    
    manifest.update({
        'end_time': None,
        'wall_time_sec': None,
    }, section='outcomes')
    
    # Create status file
    open(status_running, 'a').close()
    
    return manifest


# Legacy function wrappers for backward compatibility
def _write_txt(path, manifest_dict):
    """Legacy function - use ManifestManager instead"""
    manifest = ManifestManager(path, auto_save=False)
    manifest.data = manifest_dict
    manifest.save()


def _touch(path):
    """Create status file and clean up others"""
    status_files = [
        path.replace(os.path.basename(path), "STATUS.FAILED"),
        path.replace(os.path.basename(path), "STATUS.RUNNING"), 
        path.replace(os.path.basename(path), "STATUS.DONE")
    ]
    
    for status_file in status_files:
        if os.path.exists(status_file):
            os.remove(status_file)
    
    open(path, "a").close()


# ---------------------- CSV Logger ----------------------

class CSVLogger:
    def __init__(
        self,
        output_path,
        fieldnames: list,
        overall_start_time,
        filename: str = "CSV_metrics.csv",
        flush_every: int = 10,
    ):
        self.filepath = os.path.join(output_path, "CSV_metrics.csv")
        self.fieldnames = fieldnames
        self.overall_start_time = overall_start_time
        self.buffer = []
        self.flush_every = flush_every
        self.counter = 0

        if config.resume_from_checkpoint:
            if not os.path.exists(self.filepath):
                main_print(f"[WARNING] Checkpoint CSV file does not exist: {self.filepath}")
                sys.exit(1)
        else:
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        """Log a row of data with automatic timestamping."""
        row = {key: kwargs.get(key, None) for key in self.fieldnames}
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["overall_elapsed"] = time.time() - self.overall_start_time
        self.buffer.append(row)
        self.counter += 1
        if self.counter >= self.flush_every:
            self.flush()

    def flush(self):
        """Write the buffered log entries to file."""
        if not self.buffer:
            return
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.buffer.clear()
        self.counter = 0

# ---------------------- Dataset handling ----------------------

class CustomPadCollator:
    def __init__(self, max_length, pad_token_id: int = -100, pad_label_id: int = -100, pad_attention_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id
        self.pad_attention_id = pad_attention_id

    def __call__(self, batch):
        batch_padded = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Track other keys
        other_keys = [k for k in batch[0].keys() if k not in batch_padded]

        for item in batch:
            length = len(item["input_ids"])
            pad_len = self.max_length - length

            batch_padded["input_ids"].append(torch.cat([
                torch.tensor(item["input_ids"]),
                torch.full((pad_len,), self.pad_token_id, dtype=torch.tensor(item["input_ids"]).dtype)
            ]))

            batch_padded["attention_mask"].append(torch.cat([
                torch.tensor(item["attention_mask"]),
                torch.full((pad_len,), self.pad_attention_id, dtype=torch.tensor(item["attention_mask"]).dtype)
            ]))

            batch_padded["labels"].append(torch.cat([
                torch.tensor(item["labels"]),
                torch.full((pad_len,), self.pad_label_id, dtype=torch.tensor(item["labels"]).dtype)
            ]))

        # Stack padded fields
        for k in ["input_ids", "attention_mask", "labels"]:
            batch_padded[k] = torch.stack(batch_padded[k])

        # Add other keys without padding (just stack as-is)
        for k in other_keys:
            values = [item[k] for item in batch]
            try:
                batch_padded[k] = torch.stack(values)
            except:
                batch_padded[k] = values  # Leave as list if not stackable

        return batch_padded


def prepare_dataset(train_ds, eval_ds):
    """Prepare datasets with distributed samplers for FSDP2 training."""

    custom_collator = CustomPadCollator(1024, pad_token_id=100277) # pad_token_id for OLmo2

    train_sampler = DistributedSampler(
        train_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=True,
        seed=config.seed,
        drop_last=True, 
    )
    test_sampler = DistributedSampler(
        eval_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=False,
        drop_last=True, 
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=custom_collator,
        num_workers=0,
        persistent_workers=False,
        drop_last=True
    )   
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=custom_collator,
        num_workers=0,
        persistent_workers=False,
        drop_last=True  
    )

    dist.barrier()
    return train_dataloader, eval_dataloader
    
class DistillDataset:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.get_dataset()

    def get_dataset(self, teacher_logprobs=False):
        """Load the base dataset."""
        if config.synthetic_data:
            dataset = datasets.load_from_disk(config.synthetic_dataset_path)
            return dataset
        else:
            dataset = datasets.load_from_disk(config.dataset_path)

        if config.dataset_type == "single":
            return {
                "train": dataset["train"].select([0]),
                "test": dataset["test"].select([0]),
            }
        elif config.dataset_type == "batch":
            return {
                "train": dataset["train"].select(range(10)),
                "test": dataset["test"].select(range(10)),
            }
        
        return dataset

    def filter_by_ids(self, dataset, domain):
        """Subset a dataset cluster for C-BTM method"""
        clustered_dataset = datasets.load_dataset(config.clustered_dataset_name, domain, split="train")
        clustered_ids = set(clustered_dataset["id"])

        def subsample_ids(sample):
            """Filter to keep only examples that are in the cluster"""
            return sample["id"] in clustered_ids

        subsampled_dataset = dataset.filter(subsample_ids, num_proc=8)

        save_path = os.path.join(config.dataset_path, f"clustered/{domain}")
        if is_main_process():
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            subsampled_dataset.save_to_disk(save_path)
        dist.barrier()
        return save_path

    def get_teacher_logprobs(self):
        """Load or generate teacher logits dataset."""
        if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
            self.cache_teacher_logprobs()

        main_print("--> Loading Teacher Logits")
        dataset = datasets.load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))
        
        return dataset

    def concatenate_logprobs_chunks(self, split_dirs: list[str]):
        """Concatenate logit chunks into a single dataset."""
        datasets_list = []
        for split_dir in split_dirs:
            chunk_paths = sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            main_print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
            datasets_list.extend(load_from_disk(p) for p in chunk_paths)
        return concatenate_datasets(datasets_list) if datasets_list else None

    def build_teacher_logprobs_dataset(self):
        """Build the final teacher logprobs dataset from chunks."""
        data_dict = {}

        for split in ["train", "test"]:
            split_dirs = sorted(
                [d for d in glob.glob(os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_*")) if os.path.isdir(d)]
            )
            split_ds = self.concatenate_logprobs_chunks(split_dirs)
            data_dict[split] = split_ds

        dataset = DatasetDict(data_dict)
        combined_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
        dataset.save_to_disk(combined_path)

        return combined_path

    def cache_teacher_logprobs(self):
        if config.ddp and not dist.is_initialized():
            dist.init_process_group("nccl")
            main_print(f"Using {torch.distributed.get_backend()} backend")

        rank = int(os.environ["LOCAL_RANK"]) if config.ddp else 0
        world_size = dist.get_world_size() if config.ddp else 1
        print(f"Rank: {rank}")
        print(f"World size: {world_size}")

        torch.cuda.set_device(rank)

        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
            # force_download=True,
        ).to(f"cuda:{rank}")
        teacher_model.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
        teacher_model.requires_grad_(False)

        main_print("\n--> Generating Teacher Logits")
        for split in ["train", "test"]:

            shard = self.dataset[split].shard(num_shards=world_size, index=dist.get_rank())
            save_dir = os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_rank{dist.get_rank()}")
            os.makedirs(save_dir, exist_ok=True)

            save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logprob_values": [], "logprob_indices": [], "start_idx": [], "end_idx": [], "id": []}
            chunk_id = 0

            batch_size = 6  # tune this to your GPU
            batch_data = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "id": [],
            }

            with torch.no_grad():
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
                for idx, sample in tqdm(enumerate(shard), total=len(shard)):
                    batch_data["input_ids"].append(torch.tensor(sample["input_ids"]))
                    batch_data["attention_mask"].append(torch.tensor(sample["attention_mask"]))
                    batch_data["labels"].append(torch.tensor(sample["labels"]))
                    batch_data["id"].append(sample["id"])
                    
                    if len(batch_data["input_ids"]) == batch_size or idx == len(shard) - 1:
                        input_ids = torch.stack(batch_data["input_ids"]).to(f"cuda:{rank}")
                        attention_mask = torch.stack(batch_data["attention_mask"]).to(f"cuda:{rank}")

                        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits.squeeze(0)  # [sample, 1024, 151000]
                        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

                        values, indices = torch.topk(logprobs, k=100, dim=-1)
                        values = values.to('cpu')                           # BF16
                        indices = indices.to(torch.int32).to('cpu')         # INT32

                        if len(values.shape) == 2:
                            start = torch.where(batch_data['labels'][0] != -100)[0]
                            start_idx = start[0].item() if len(start) != 0 else 0

                            end = torch.where(batch_data['input_ids'][0] == tokenizer.pad_token_id)[0]
                            end_idx = end[0].item() if len(end) != 0 else len(batch_data['input_ids'][0]) - 1

                            save_ds["id"].append(batch_data["id"][0])
                            save_ds["input_ids"].append(batch_data["input_ids"][0][:end_idx])
                            save_ds["attention_mask"].append(batch_data["attention_mask"][0][:end_idx])
                            save_ds["labels"].append(batch_data["labels"][0][:end_idx])
                            save_ds["logprob_values"].append(values[start_idx:end_idx])
                            save_ds["logprob_indices"].append(indices[start_idx:end_idx])
                            save_ds["start_idx"].append(start_idx)
                            save_ds["end_idx"].append(end_idx)
                        else:
                            for b in range(values.size(0)):
                                start = torch.where(batch_data['labels'][b] != -100)[0] 
                                if len(start) == 0:
                                    continue
                                start_idx = start[0].item()
                                
                                end = torch.where(batch_data['input_ids'][b] == tokenizer.pad_token_id)[0]
                                end_idx = end[0].item() if len(end) != 0 else len(batch_data['input_ids'][b]) - 1
                                
                                save_ds["id"].append(batch_data["id"][b])
                                save_ds["input_ids"].append(batch_data["input_ids"][b][:end_idx].tolist())
                                save_ds["attention_mask"].append(batch_data["attention_mask"][b][:end_idx].tolist())
                                save_ds["labels"].append(batch_data["labels"][b][:end_idx].tolist())
                                save_ds["logprob_values"].append(values[b][start_idx:end_idx].tolist())
                                save_ds["logprob_indices"].append(indices[b][start_idx:end_idx].tolist())
                                save_ds["start_idx"].append(start_idx)
                                save_ds["end_idx"].append(end_idx)

                        batch_data = {"input_ids": [], "attention_mask": [], "labels": [], "id": []}

                        if (idx + 1) % 3200 < batch_size or idx == len(shard) - 1:
                            main_print(f"--> [{split}] Saving chunk {chunk_id} with {len(save_ds['input_ids'])} samples")

                            save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            Dataset.from_dict(save_ds).save_to_disk(save_path)

                            save_ds = {
                                "input_ids": [],
                                "attention_mask": [],
                                "labels": [],
                                "logprob_values": [],
                                "logprob_indices": [],
                                "start_idx": [],
                                "end_idx": [],
                                "id": [],
                            }
                            chunk_id += 1

                    # if (idx + 1) % 3200 == 0:
                    #     break

            if save_ds["input_ids"]:
                print(f"--> [{split}] Saving final chunk {chunk_id} with {len(save_ds['input_ids'])} samples")
                save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                Dataset.from_dict(save_ds).save_to_disk(save_path)
        
        dist.barrier() if config.ddp else None

        if _get_rank() == 0:
            self.build_teacher_logprobs_dataset()
        
        main_print("\n--> Generation Done")
        dist.barrier() if config.ddp else None

# ---------------------- Main execution for testing ----------------------
if __name__ == "__main__":
    
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset()
    teacher_logits = dataClass.cache_teacher_logprobs()

