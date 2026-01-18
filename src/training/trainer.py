"""
Training Loop for DETR
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Dict, Optional

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import os
import signal
import sys
from pathlib import Path
import time
import gc
import sys  # Ensure sys is imported for flush()
from src.training.evaluator import Evaluator
from src.training.adaptive_optimizer import AdaptiveOptimizer

# Mixed precision training
from torch.amp import autocast, GradScaler

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# MLflow tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# MLflow debugging flag
MLFLOW_DEBUG = True  # Set to True to enable detailed MLflow error logging


class MLflowDebugger:
    """Comprehensive MLflow operation debugging"""
    
    def __init__(self, tracking_uri, run_id=None, experiment_id=None):
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.operation_count = 0
        
    def _get_timestamp(self):
        """Get formatted timestamp"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f}PB"
    
    def check_filesystem_health(self, path=None):
        """Check disk space, permissions, path validity"""
        health = {
            'disk_total': 0,
            'disk_free': 0,
            'disk_used': 0,
            'disk_percent': 0,
            'path_exists': False,
            'path_writable': False,
            'path_readable': False,
            'fs_type': None,
            'mount_point': None
        }
        
        try:
            import shutil
            import stat
            
            # Get disk usage for the path or current directory
            check_path = path if path else os.getcwd()
            statvfs = os.statvfs(check_path)
            
            # Calculate disk space
            total_bytes = statvfs.f_frsize * statvfs.f_blocks
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            used_bytes = total_bytes - free_bytes
            percent_used = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
            
            health['disk_total'] = total_bytes
            health['disk_free'] = free_bytes
            health['disk_used'] = used_bytes
            health['disk_percent'] = percent_used
            
            # Check path permissions
            if path and os.path.exists(path):
                health['path_exists'] = True
                health['path_readable'] = os.access(path, os.R_OK)
                health['path_writable'] = os.access(path, os.W_OK)
            
            # Try to get filesystem info
            try:
                if PSUTIL_AVAILABLE:
                    disk = psutil.disk_usage(check_path)
                    health['disk_total'] = disk.total
                    health['disk_free'] = disk.free
                    health['disk_used'] = disk.used
                    health['disk_percent'] = disk.percent
            except:
                pass
                
        except Exception as e:
            if MLFLOW_DEBUG:
                print(f"[MLFLOW DEBUG] Error checking filesystem health: {e}")
        
        return health
    
    def check_mlflow_backend_health(self):
        """Verify MLflow backend is accessible"""
        health = {
            'backend_accessible': False,
            'experiment_accessible': False,
            'run_accessible': False,
            'run_active': False,
            'tracking_uri': self.tracking_uri,
            'error': None
        }
        
        if not MLFLOW_AVAILABLE:
            health['error'] = "MLflow not available"
            return health
        
        try:
            # Check tracking URI
            current_uri = mlflow.get_tracking_uri()
            health['backend_accessible'] = (current_uri == self.tracking_uri or self.tracking_uri in current_uri)
            
            # Check experiment
            if self.experiment_id:
                try:
                    experiment = mlflow.get_experiment(self.experiment_id)
                    health['experiment_accessible'] = experiment is not None
                except:
                    health['experiment_accessible'] = False
            
            # Check run
            if self.run_id:
                try:
                    run = mlflow.get_run(self.run_id)
                    health['run_accessible'] = run is not None
                    health['run_active'] = run.info.status == "RUNNING"
                except:
                    health['run_accessible'] = False
                    health['run_active'] = False
                    
        except Exception as e:
            health['error'] = str(e)
            if MLFLOW_DEBUG:
                print(f"[MLFLOW DEBUG] Backend health check error: {e}")
        
        return health
    
    def log_operation_start(self, operation_name, **kwargs):
        """Log before MLflow operation"""
        self.operation_count += 1
        timestamp = self._get_timestamp()
        
        # Get filesystem health
        fs_health = self.check_filesystem_health()
        
        # Build log message
        details = []
        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
            details.append(f"file={file_path}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                details.append(f"size={self._format_size(file_size)}")
        if 'artifact_path' in kwargs:
            details.append(f"artifact_path={kwargs['artifact_path']}")
        if 'metric_name' in kwargs:
            details.append(f"metric={kwargs['metric_name']}")
            details.append(f"value={kwargs.get('value', 'N/A')}")
        if 'step' in kwargs:
            details.append(f"step={kwargs['step']}")
        
        details.append(f"disk_free={self._format_size(fs_health['disk_free'])}")
        details.append(f"disk_used={fs_health['disk_percent']:.1f}%")
        
        log_msg = f"[MLFLOW DEBUG] [{timestamp}] [{operation_name}] [START] {' '.join(details)}"
        print(log_msg, flush=True)
        
        return {
            'start_time': time.time(),
            'operation_name': operation_name,
            'fs_health': fs_health,
            'kwargs': kwargs
        }
    
    def log_operation_success(self, operation_name, duration, context=None, **kwargs):
        """Log after successful operation"""
        timestamp = self._get_timestamp()
        
        # Get filesystem health after operation
        fs_health = self.check_filesystem_health()
        
        details = []
        details.append(f"duration={duration:.3f}s")
        
        if context and 'file_path' in context.get('kwargs', {}):
            file_path = context['kwargs']['file_path']
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                details.append(f"file_size={self._format_size(file_size)}")
        
        details.append(f"disk_free={self._format_size(fs_health['disk_free'])}")
        details.append(f"disk_used={fs_health['disk_percent']:.1f}%")
        
        log_msg = f"[MLFLOW DEBUG] [{timestamp}] [{operation_name}] [SUCCESS] {' '.join(details)}"
        print(log_msg, flush=True)
    
    def log_operation_error(self, operation_name, error, context=None, **kwargs):
        """Log detailed error information"""
        timestamp = self._get_timestamp()
        
        # Get filesystem health
        fs_health = self.check_filesystem_health()
        
        # Get backend health
        backend_health = self.check_mlflow_backend_health()
        
        # Extract error details
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Check for MLflow-specific error codes
        mlflow_error_code = None
        if hasattr(error, 'error_code'):
            mlflow_error_code = error.error_code
        elif 'INTERNAL_ERROR' in error_msg or 'INTERNAL_ERROR' in error_type:
            mlflow_error_code = 'INTERNAL_ERROR'
        
        # Get full traceback
        import traceback
        tb_str = traceback.format_exc()
        
        # Build log message
        details = []
        details.append(f"error_type={error_type}")
        if mlflow_error_code:
            details.append(f"mlflow_error_code={mlflow_error_code}")
        details.append(f"error_msg={error_msg}")
        
        if context:
            if 'file_path' in context.get('kwargs', {}):
                details.append(f"file={context['kwargs']['file_path']}")
            if 'artifact_path' in context.get('kwargs', {}):
                details.append(f"artifact_path={context['kwargs']['artifact_path']}")
            if 'start_time' in context:
                elapsed = time.time() - context['start_time']
                details.append(f"elapsed={elapsed:.3f}s")
        
        details.append(f"disk_free={self._format_size(fs_health['disk_free'])}")
        details.append(f"disk_used={fs_health['disk_percent']:.1f}%")
        details.append(f"backend_accessible={backend_health['backend_accessible']}")
        details.append(f"run_active={backend_health['run_active']}")
        
        # OSError specific details
        if isinstance(error, (OSError, IOError)):
            if hasattr(error, 'errno'):
                details.append(f"errno={error.errno}")
            if hasattr(error, 'strerror'):
                details.append(f"strerror={error.strerror}")
        
        log_msg = f"[MLFLOW DEBUG] [{timestamp}] [{operation_name}] [ERROR] {' '.join(details)}"
        print(log_msg, flush=True)
        
        # Print traceback if debug enabled
        if MLFLOW_DEBUG:
            print(f"[MLFLOW DEBUG] Full traceback:\n{tb_str}", flush=True)
        
        # Print filesystem details
        if MLFLOW_DEBUG:
            print(f"[MLFLOW DEBUG] Filesystem state: total={self._format_size(fs_health['disk_total'])}, "
                  f"free={self._format_size(fs_health['disk_free'])}, "
                  f"used={fs_health['disk_percent']:.1f}%", flush=True)
        
        # Print backend state
        if MLFLOW_DEBUG:
            print(f"[MLFLOW DEBUG] Backend state: uri={backend_health['tracking_uri']}, "
                  f"experiment_accessible={backend_health['experiment_accessible']}, "
                  f"run_accessible={backend_health['run_accessible']}, "
                  f"run_active={backend_health['run_active']}", flush=True)
            if backend_health['error']:
                print(f"[MLFLOW DEBUG] Backend error: {backend_health['error']}", flush=True)


class Trainer:
    """DETR Trainer"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict, device: torch.device,
                 writer: Optional = None, mlflow_run=None):
        """
        Initialize trainer
        
        Args:
            model: DETR model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            writer: TensorBoard writer (optional)
            mlflow_run: MLflow run object (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.writer = writer
        self.mlflow_run = mlflow_run
        self.use_mlflow = mlflow_run is not None and MLFLOW_AVAILABLE
        self.mlflow_failure_count = 0  # Track consecutive MLflow failures
        self.mlflow_max_failures = 5  # Disable MLflow after this many consecutive failures
        # Disable model logging by default (causes INTERNAL_ERROR) - can be enabled via config
        self.mlflow_model_logging_enabled = config.get('logging', {}).get('mlflow_log_models', False)
        if not self.mlflow_model_logging_enabled:
            print("[MLFLOW] Model artifact logging disabled (set mlflow_log_models: true in config to enable)")
        
        # Initialize MLflow debugger (always initialize, even if MLflow is disabled)
        tracking_uri = config.get('logging', {}).get('mlflow_tracking_uri', 'file:./mlruns')
        if self.use_mlflow and mlflow_run:
            run_id = mlflow_run.info.run_id if hasattr(mlflow_run, 'info') else None
            experiment_id = mlflow_run.info.experiment_id if hasattr(mlflow_run, 'info') else None
            self.mlflow_debugger = MLflowDebugger(tracking_uri, run_id, experiment_id)
        else:
            # Initialize with None values if MLflow is not available
            self.mlflow_debugger = MLflowDebugger(tracking_uri, None, None)
        
        # Setup mixed precision training (AMP)
        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")
        
        # Setup gradient accumulation
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            print(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
        
        # Memory cleanup frequency
        self.memory_cleanup_frequency = config['training'].get('memory_cleanup_frequency', 10)
        
        # Setup adaptive optimizer (if enabled)
        self.adaptive_optimizer = None
        if config['training'].get('adaptive_optimization', False):
            self.adaptive_optimizer = AdaptiveOptimizer(
                initial_num_workers=config['dataset']['num_workers'],
                initial_prefetch_factor=config['dataset'].get('prefetch_factor', 2),
                target_gpu_utilization=config['training'].get('target_gpu_utilization', 0.85),
                max_ram_usage=config['training'].get('max_ram_usage', 0.80),
                adjustment_interval=config['training'].get('adaptive_adjustment_interval', 50)
            )
            print("Adaptive optimization enabled - monitoring resource usage")
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Setup learning rate scheduler
        warmup_epochs = config['lr_schedule']['warmup_epochs']
        num_epochs = config['training']['num_epochs']
        
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs * len(train_loader)
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=(num_epochs - warmup_epochs) * len(train_loader),
                eta_min=config['lr_schedule'].get('min_lr', 1e-6)
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs * len(train_loader)]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs * len(train_loader),
                eta_min=config['lr_schedule'].get('min_lr', 1e-6)
            )
        
        # Setup evaluator
        self.evaluator = Evaluator(config['evaluation'])
        
        # Training state
        self.best_map = 0.0
        self.global_step = 0
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n\nReceived signal {signum}. Saving checkpoint before exit...")
            self.interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        memory_info = {}
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info['ram_gb'] = process.memory_info().rss / (1024 ** 3)
        if torch.cuda.is_available():
            memory_info['gpu_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        return memory_info
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        print(f"[VERBOSE] train_epoch({epoch}) called", flush=True)
        
        self.model.train()
        print(f"[VERBOSE] Model set to train mode", flush=True)
        
        total_loss = 0.0
        num_batches = 0
        
        print_freq = self.config['logging']['print_frequency']
        
        # Initialize gradient accumulation
        accumulation_loss = 0.0
        accumulation_loss_components = {}  # Track individual loss components
        accumulation_count = 0
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        print(f"[VERBOSE] Gradients zeroed, about to iterate over train_loader (len={len(self.train_loader)})", flush=True)
        print(f"[VERBOSE] Getting iterator from train_loader...", flush=True)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if batch_idx == 0:
                print(f"[VERBOSE] Starting batch 0 at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            elif batch_idx % 10 == 0:
                print(f"[VERBOSE] Processing batch {batch_idx}/{len(self.train_loader)}", flush=True)
            
            # Track data loading time (approximate)
            data_load_start = time.time()
            if batch_idx == 0:
                print(f"[VERBOSE] Batch {batch_idx}: Data loaded, moving to device...", flush=True)
            
            # Move to device
            # DETR expects list of images, not batched tensor
            if isinstance(images, torch.Tensor):
                images = [img.to(self.device) for img in images]
            else:
                images = [img.to(self.device) for img in images]
            
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            data_load_time = time.time() - data_load_start
            if batch_idx == 0:
                print(f"[VERBOSE] Batch {batch_idx}: Moved to device in {data_load_time:.3f}s, about to forward pass...", flush=True)
            
            # Convert images to channels-last if enabled
            if hasattr(self.model, 'memory_format') and self.model.memory_format == torch.channels_last:
                images = [img.to(memory_format=torch.channels_last) if isinstance(img, torch.Tensor) else img 
                         for img in images]
            
            # Track GPU processing time
            gpu_start = time.time()
            if batch_idx == 0:
                print(f"[VERBOSE] Batch {batch_idx}: Starting forward pass (AMP={self.use_amp})...", flush=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    if batch_idx == 0:
                        print(f"[VERBOSE] Batch {batch_idx}: Calling model.forward()...", flush=True)
                    loss_dict = self.model(images, targets)
                    if batch_idx == 0:
                        print(f"[VERBOSE] Batch {batch_idx}: model.forward() completed", flush=True)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Scale loss by accumulation steps
                scaled_loss = losses / self.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(scaled_loss).backward()
            else:
                # Standard precision
                if batch_idx == 0:
                    print(f"[VERBOSE] Batch {batch_idx}: Calling model.forward() (standard precision)...", flush=True)
                loss_dict = self.model(images, targets)
                if batch_idx == 0:
                    print(f"[VERBOSE] Batch {batch_idx}: model.forward() completed", flush=True)
                losses = sum(loss for loss in loss_dict.values())
                
                # Scale loss by accumulation steps
                scaled_loss = losses / self.gradient_accumulation_steps
                
                # Backward pass
                scaled_loss.backward()
            
            gpu_processing_time = time.time() - gpu_start
            
            # Record timing for adaptive optimizer
            if self.adaptive_optimizer:
                self.adaptive_optimizer.record_batch_timing(data_load_time, gpu_processing_time)
            
            # Accumulate loss and individual components
            accumulation_loss += losses.item()
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in accumulation_loss_components:
                    accumulation_loss_components[loss_name] = 0.0
                accumulation_loss_components[loss_name] += loss_value.item()
            accumulation_count += 1
            
            # Only step optimizer after accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients after step
                self.optimizer.zero_grad()
            
            # Scheduler step (every batch, not every accumulation step)
            self.scheduler.step()
            
            # Adaptive optimization check
            if self.adaptive_optimizer:
                adjustment = self.adaptive_optimizer.adjust_parameters(batch_idx)
                if adjustment:
                    print(f"\n🔧 Adaptive Optimization Adjustment (batch {batch_idx}):")
                    for key, value in adjustment.items():
                        if key != 'metrics':
                            print(f"   {key}: {value}")
                    if 'metrics' in adjustment:
                        m = adjustment['metrics']
                        print(f"   GPU util: {m['avg_gpu_utilization']:.1%}, RAM: {m['avg_ram_usage']:.1%}")
                        print(f"   Data load: {m['avg_data_loading_time']:.3f}s, GPU process: {m['avg_gpu_processing_time']:.3f}s")
                    
                    # Log to MLflow (with error handling and debugging)
                    if self.use_mlflow:
                        try:
                            for key, value in adjustment.items():
                                if key != 'metrics':
                                    if self.mlflow_debugger and self.use_mlflow:
                                        context = self.mlflow_debugger.log_operation_start(
                                            'log_metric',
                                            metric_name=f'adaptive_{key}',
                                            value=value,
                                            step=self.global_step
                                        )
                                    start_time = time.time()
                                    mlflow.log_metric(f'adaptive_{key}', value, step=self.global_step)
                                    if self.mlflow_debugger and self.use_mlflow:
                                        self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                        except Exception as mlflow_error:
                            # Log detailed error information
                            if self.mlflow_debugger and self.use_mlflow:
                                self.mlflow_debugger.log_operation_error(
                                    'log_metric',
                                    mlflow_error,
                                    context={'operation': 'adaptive_optimization', 'step': self.global_step}
                                )
                            # Don't let MLflow errors stop training
                            if MLFLOW_DEBUG:
                                print(f"[MLFLOW DEBUG] Adaptive MLflow logging failed (non-blocking): {type(mlflow_error).__name__}: {mlflow_error}")
                            pass  # Silently ignore adaptive optimization MLflow errors
            
            # Memory cleanup
            del images, targets, loss_dict, losses
            if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Logging (only on accumulation boundaries or last batch)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_loss = accumulation_loss / accumulation_count if accumulation_count > 0 else 0.0
                total_loss += avg_loss * accumulation_count
                num_batches += accumulation_count
                self.global_step += 1
                
                # Memory monitoring
                memory_info = {}
                if batch_idx % (print_freq * self.gradient_accumulation_steps) == 0:
                    memory_info = self._get_memory_usage()
                    if memory_info:
                        mem_str = ", ".join([f"{k}: {v:.2f}GB" for k, v in memory_info.items()])
                        print(f"Memory: {mem_str}")
                
                # Reduced logging frequency for less I/O overhead
                log_every_n_steps = self.config['logging'].get('log_every_n_steps', 10)
                
                if batch_idx % (print_freq * self.gradient_accumulation_steps) == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
                
                # Less frequent TensorBoard logging
                if self.writer and self.global_step % log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Train/Loss', avg_loss, self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    if memory_info:
                        for key, value in memory_info.items():
                            self.writer.add_scalar(f'Memory/{key}', value, self.global_step)
                
                # MLflow logging (with error handling and retry logic to prevent training interruption)
                if self.use_mlflow and self.global_step % log_every_n_steps == 0:
                    # Circuit breaker: disable MLflow if too many consecutive failures
                    if self.mlflow_failure_count >= self.mlflow_max_failures:
                        if self.global_step % (log_every_n_steps * 50) == 0:  # Only warn occasionally
                            print(f"WARNING: MLflow disabled due to {self.mlflow_failure_count} consecutive failures")
                        self.use_mlflow = False
                        continue
                    
                    max_retries = 2
                    retry_count = 0
                    logged_successfully = False
                    
                    while retry_count < max_retries and not logged_successfully:
                        context = None
                        try:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            
                            # Pre-operation validation
                            if self.mlflow_debugger and self.use_mlflow:
                                # Validate metrics before logging
                                if not (isinstance(avg_loss, (int, float)) and not (torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)))):
                                    print(f"[MLFLOW DEBUG] WARNING: Invalid loss value: {avg_loss}")
                                if not (isinstance(current_lr, (int, float)) and not (torch.isnan(torch.tensor(current_lr)) or torch.isinf(torch.tensor(current_lr)))):
                                    print(f"[MLFLOW DEBUG] WARNING: Invalid LR value: {current_lr}")
                            
                            # Log train_loss metric
                            if self.mlflow_debugger and self.use_mlflow:
                                context = self.mlflow_debugger.log_operation_start(
                                    'log_metric',
                                    metric_name='train_loss',
                                    value=avg_loss,
                                    step=self.global_step
                                )
                            start_time = time.time()
                            mlflow.log_metric('train_loss', avg_loss, step=self.global_step)
                            if self.mlflow_debugger and self.use_mlflow:
                                self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                            
                            # Log learning_rate metric
                            if self.mlflow_debugger and self.use_mlflow:
                                context = self.mlflow_debugger.log_operation_start(
                                    'log_metric',
                                    metric_name='learning_rate',
                                    value=current_lr,
                                    step=self.global_step
                                )
                            start_time = time.time()
                            mlflow.log_metric('learning_rate', current_lr, step=self.global_step)
                            if self.mlflow_debugger and self.use_mlflow:
                                self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                            
                            # Log individual loss components
                            if accumulation_loss_components:
                                for loss_name, loss_sum in accumulation_loss_components.items():
                                    avg_component_loss = loss_sum / accumulation_count if accumulation_count > 0 else 0.0
                                    if self.mlflow_debugger and self.use_mlflow:
                                        context = self.mlflow_debugger.log_operation_start(
                                            'log_metric',
                                            metric_name=f'train_{loss_name}',
                                            value=avg_component_loss,
                                            step=self.global_step
                                        )
                                    start_time = time.time()
                                    mlflow.log_metric(f'train_{loss_name}', avg_component_loss, step=self.global_step)
                                    if self.mlflow_debugger and self.use_mlflow:
                                        self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                            
                            if memory_info:
                                for key, value in memory_info.items():
                                    if self.mlflow_debugger and self.use_mlflow:
                                        context = self.mlflow_debugger.log_operation_start(
                                            'log_metric',
                                            metric_name=f'memory_{key}',
                                            value=value,
                                            step=self.global_step
                                        )
                                    start_time = time.time()
                                    mlflow.log_metric(f'memory_{key}', value, step=self.global_step)
                                    if self.mlflow_debugger and self.use_mlflow:
                                        self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                            
                            logged_successfully = True
                            self.mlflow_failure_count = 0  # Reset failure count on success
                            
                        except Exception as mlflow_error:
                            retry_count += 1
                            
                            # Log detailed error information
                            if self.mlflow_debugger and self.use_mlflow:
                                self.mlflow_debugger.log_operation_error(
                                    'log_metric',
                                    mlflow_error,
                                    context=context,
                                    retry_count=retry_count,
                                    max_retries=max_retries
                                )
                            
                            error_type = type(mlflow_error).__name__
                            error_msg = f"MLflow logging failed (non-blocking) at step {self.global_step}, attempt {retry_count}/{max_retries}: {error_type}: {mlflow_error}"
                            
                            # Increment failure count
                            self.mlflow_failure_count += 1
                            
                            # Check if it's a recoverable error
                            recoverable_errors = ('RestException', 'ConnectionError', 'Timeout', 'INTERNAL_ERROR')
                            is_recoverable = any(err in str(mlflow_error) or err in error_type for err in recoverable_errors)
                            
                            if retry_count < max_retries and is_recoverable:
                                if MLFLOW_DEBUG:
                                    print(f"[MLFLOW DEBUG] {error_msg} - Retrying...")
                                time.sleep(0.5)  # Brief delay before retry
                            else:
                                # Don't let MLflow errors stop training
                                if self.global_step % (log_every_n_steps * 10) == 0:  # Only print every 10th failure
                                    print(f"WARNING: {error_msg} (failure count: {self.mlflow_failure_count}/{self.mlflow_max_failures})")
                                # Disable MLflow if too many failures
                                if self.mlflow_failure_count >= self.mlflow_max_failures:
                                    print(f"WARNING: MLflow disabled after {self.mlflow_failure_count} consecutive failures. Training continues without MLflow.")
                                    self.use_mlflow = False
                                break
                
                # Reset accumulation
                accumulation_loss = 0.0
                accumulation_loss_components = {}
                accumulation_count = 0
        
        # Handle remaining gradients if last batch didn't complete accumulation cycle
        if accumulation_count > 0:
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step with remaining accumulated gradients
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Log final accumulation
            avg_loss = accumulation_loss / accumulation_count
            total_loss += avg_loss * accumulation_count
            num_batches += accumulation_count
            self.global_step += 1
            
            # Reset accumulation components for next epoch
            accumulation_loss_components = {}
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[VERBOSE] train_epoch({epoch}) finished: {num_batches} batches, avg_loss={avg_loss:.4f}", flush=True)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # DETR expects list of images
                if isinstance(images, torch.Tensor):
                    images = [img.to(self.device) for img in images]
                else:
                    images = [img.to(self.device) for img in images]
                
                # Get predictions (in eval mode, model returns predictions)
                # Use autocast for validation too if AMP is enabled
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Store for evaluation
                # Convert target labels from 1-based (1=player, 2=ball) to 0-based (0=player, 1=ball)
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    # Convert target labels: 1→0 (player), 2→1 (ball)
                    target_0based = target.copy()
                    if 'labels' in target_0based and len(target_0based['labels']) > 0:
                        target_0based['labels'] = target_0based['labels'] - 1
                    all_predictions.append(output)
                    all_targets.append(target_0based)
                
                # Memory cleanup during validation
                del images, targets, outputs
                if (batch_idx + 1) % self.memory_cleanup_frequency == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Evaluate
        eval_metrics = self.evaluator.evaluate(all_predictions, all_targets)
        map_score = eval_metrics['map']
        
        print(f"Validation mAP: {map_score:.4f}")
        print(f"Validation Precision: {eval_metrics['precision']:.4f}")
        print(f"Validation Recall: {eval_metrics['recall']:.4f}")
        print(f"Validation F1: {eval_metrics['f1']:.4f}")
        print(f"\nPer-Class Metrics (IoU 0.5):")
        print(f"  Player - mAP@0.5: {eval_metrics.get('player_map_05', eval_metrics.get('player_map', 0.0)):.4f}, Precision: {eval_metrics.get('player_precision_05', eval_metrics.get('player_precision', 0.0)):.4f}, Recall: {eval_metrics.get('player_recall_05', eval_metrics.get('player_recall', 0.0)):.4f}, F1: {eval_metrics.get('player_f1', 0.0):.4f}")
        print(f"  Ball   - mAP@0.5: {eval_metrics.get('ball_map_05', eval_metrics.get('ball_map', 0.0)):.4f}, Precision: {eval_metrics.get('ball_precision_05', eval_metrics.get('ball_precision', 0.0)):.4f}, Recall: {eval_metrics.get('ball_recall_05', eval_metrics.get('ball_recall', 0.0)):.4f}, F1: {eval_metrics.get('ball_f1', 0.0):.4f}")
        print(f"\nPer-Class Metrics (IoU 0.75):")
        print(f"  Player - mAP@0.75: {eval_metrics.get('player_map_75', 0.0):.4f}")
        print(f"  Ball   - mAP@0.75: {eval_metrics.get('ball_map_75', 0.0):.4f}")
        if 'ball_avg_predictions_per_image' in eval_metrics:
            print(f"\nBall Detection:")
            print(f"  Avg predictions per image with balls: {eval_metrics['ball_avg_predictions_per_image']:.2f}")
            print(f"  Images with balls: {eval_metrics.get('images_with_balls', 0)}")
        
        if self.writer:
            self.writer.add_scalar('Val/mAP', map_score, epoch)
            self.writer.add_scalar('Val/Precision', eval_metrics['precision'], epoch)
            self.writer.add_scalar('Val/Recall', eval_metrics['recall'], epoch)
            self.writer.add_scalar('Val/F1', eval_metrics['f1'], epoch)
            # Per-class metrics
            self.writer.add_scalar('Val/Player_mAP', eval_metrics['player_map'], epoch)
            self.writer.add_scalar('Val/Player_Precision', eval_metrics['player_precision'], epoch)
            self.writer.add_scalar('Val/Player_Recall', eval_metrics['player_recall'], epoch)
            self.writer.add_scalar('Val/Player_F1', eval_metrics['player_f1'], epoch)
            self.writer.add_scalar('Val/Ball_mAP', eval_metrics['ball_map'], epoch)
            self.writer.add_scalar('Val/Ball_Precision', eval_metrics['ball_precision'], epoch)
            self.writer.add_scalar('Val/Ball_Recall', eval_metrics['ball_recall'], epoch)
            self.writer.add_scalar('Val/Ball_F1', eval_metrics['ball_f1'], epoch)
        
        # MLflow logging for validation (with error handling and retry logic)
        if self.use_mlflow:
            # Circuit breaker: skip if too many failures
            if self.mlflow_failure_count >= self.mlflow_max_failures:
                return map_score
            
            max_retries = 2
            retry_count = 0
            logged_successfully = False
            
            while retry_count < max_retries and not logged_successfully:
                context = None
                try:
                    # Pre-operation validation - check backend health
                    if self.mlflow_debugger and self.use_mlflow:
                        backend_health = self.mlflow_debugger.check_mlflow_backend_health()
                        if not backend_health['backend_accessible']:
                            print(f"[MLFLOW DEBUG] WARNING: Backend not accessible before validation logging")
                        if not backend_health['run_active']:
                            print(f"[MLFLOW DEBUG] WARNING: Run not active before validation logging")
                    
                    # Overall metrics - log with debugging
                    metrics_to_log = [
                        ('val_map', map_score),
                        ('val_precision', eval_metrics['precision']),
                        ('val_recall', eval_metrics['recall']),
                        ('val_f1', eval_metrics['f1']),
                        ('val_player_map_05', eval_metrics['player_map_05']),
                        ('val_player_precision_05', eval_metrics['player_precision_05']),
                        ('val_player_recall_05', eval_metrics['player_recall_05']),
                        ('val_player_f1', eval_metrics['player_f1']),
                        ('val_player_map_75', eval_metrics['player_map_75']),
                        ('val_ball_map_05', eval_metrics['ball_map_05']),
                        ('val_ball_precision_05', eval_metrics['ball_precision_05']),
                        ('val_ball_recall_05', eval_metrics['ball_recall_05']),
                        ('val_ball_f1', eval_metrics['ball_f1']),
                        ('val_ball_map_75', eval_metrics['ball_map_75']),
                        ('val_ball_avg_predictions_per_image', eval_metrics['ball_avg_predictions_per_image']),
                        ('val_images_with_balls', eval_metrics['images_with_balls']),
                        ('val_player_map', eval_metrics.get('player_map', eval_metrics['player_map_05'])),
                        ('val_player_precision', eval_metrics.get('player_precision', eval_metrics['player_precision_05'])),
                        ('val_player_recall', eval_metrics.get('player_recall', eval_metrics['player_recall_05'])),
                        ('val_ball_map', eval_metrics.get('ball_map', eval_metrics['ball_map_05'])),
                        ('val_ball_precision', eval_metrics.get('ball_precision', eval_metrics['ball_precision_05'])),
                        ('val_ball_recall', eval_metrics.get('ball_recall', eval_metrics['ball_recall_05'])),
                    ]
                    
                    for metric_name, metric_value in metrics_to_log:
                        # Validate metric value
                        if self.mlflow_debugger and self.use_mlflow:
                            if not isinstance(metric_value, (int, float)) or (isinstance(metric_value, float) and (torch.isnan(torch.tensor(metric_value)) or torch.isinf(torch.tensor(metric_value)))):
                                print(f"[MLFLOW DEBUG] WARNING: Invalid metric value for {metric_name}: {metric_value}")
                        
                        if self.mlflow_debugger and self.use_mlflow:
                            context = self.mlflow_debugger.log_operation_start(
                                'log_metric',
                                metric_name=metric_name,
                                value=metric_value,
                                step=epoch
                            )
                        start_time = time.time()
                        mlflow.log_metric(metric_name, metric_value, step=epoch)
                        if self.mlflow_debugger and self.use_mlflow:
                            self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
                    
                    # Goal tracking - log goal achievement status
                    self._log_goals_to_mlflow(eval_metrics, epoch)
                    
                    logged_successfully = True
                    self.mlflow_failure_count = 0  # Reset failure count on success
                    
                except Exception as mlflow_error:
                    retry_count += 1
                    
                    # Log detailed error information
                    if self.mlflow_debugger and self.use_mlflow:
                        self.mlflow_debugger.log_operation_error(
                            'log_metric',
                            mlflow_error,
                            context=context,
                            retry_count=retry_count,
                            max_retries=max_retries,
                            epoch=epoch
                        )
                    
                    error_type = type(mlflow_error).__name__
                    error_msg = f"MLflow validation logging failed (non-blocking) at epoch {epoch}, attempt {retry_count}/{max_retries}: {error_type}: {mlflow_error}"
                    
                    # Increment failure count
                    self.mlflow_failure_count += 1
                    
                    # Check if recoverable
                    recoverable = any(err in str(mlflow_error) or err in error_type for err in ('RestException', 'ConnectionError', 'Timeout', 'INTERNAL_ERROR'))
                    
                    if retry_count < max_retries and recoverable:
                        if MLFLOW_DEBUG:
                            print(f"[MLFLOW DEBUG] {error_msg} - Retrying...")
                        time.sleep(1)
                    else:
                        print(f"WARNING: {error_msg} (failure count: {self.mlflow_failure_count}/{self.mlflow_max_failures})")
                        # Disable MLflow if too many failures
                        if self.mlflow_failure_count >= self.mlflow_max_failures:
                            print(f"WARNING: MLflow disabled after {self.mlflow_failure_count} consecutive failures. Training continues without MLflow.")
                            self.use_mlflow = False
                        break
        
        # Final cleanup after validation
        del all_predictions, all_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return map_score
    
    def _log_goals_to_mlflow(self, eval_metrics: Dict[str, float], epoch: int):
        """
        Log goal tracking metrics to MLflow
        
        Args:
            eval_metrics: Dictionary of evaluation metrics
            epoch: Current epoch number
        """
        if not self.use_mlflow:
            return
        
        # Define goals
        goals = {
            # Player goals
            'goal_player_recall_05': 0.95,
            'goal_player_precision_05': 0.80,
            'goal_player_map_05': 0.85,
            'goal_player_map_75': 0.70,
            
            # Ball goals
            'goal_ball_recall_05': 0.80,
            'goal_ball_precision_05': 0.70,
            'goal_ball_map_05': 0.70,
            'goal_ball_avg_predictions_per_image': 1.0,  # At least 1 prediction per image with balls
        }
        
        # Log goal achievement status (1.0 = achieved, 0.0 = not achieved)
        goal_achievements = {}
        
        # Player goals
        player_recall_05 = eval_metrics.get('player_recall_05', 0.0)
        player_precision_05 = eval_metrics.get('player_precision_05', 0.0)
        player_map_05 = eval_metrics.get('player_map_05', 0.0)
        player_map_75 = eval_metrics.get('player_map_75', 0.0)
        
        goal_achievements['goal_player_recall_05_achieved'] = 1.0 if player_recall_05 >= goals['goal_player_recall_05'] else 0.0
        goal_achievements['goal_player_precision_05_achieved'] = 1.0 if player_precision_05 >= goals['goal_player_precision_05'] else 0.0
        goal_achievements['goal_player_map_05_achieved'] = 1.0 if player_map_05 >= goals['goal_player_map_05'] else 0.0
        goal_achievements['goal_player_map_75_achieved'] = 1.0 if player_map_75 >= goals['goal_player_map_75'] else 0.0
        
        # Ball goals
        ball_recall_05 = eval_metrics.get('ball_recall_05', 0.0)
        ball_precision_05 = eval_metrics.get('ball_precision_05', 0.0)
        ball_map_05 = eval_metrics.get('ball_map_05', 0.0)
        ball_avg_preds = eval_metrics.get('ball_avg_predictions_per_image', 0.0)
        
        goal_achievements['goal_ball_recall_05_achieved'] = 1.0 if ball_recall_05 >= goals['goal_ball_recall_05'] else 0.0
        goal_achievements['goal_ball_precision_05_achieved'] = 1.0 if ball_precision_05 >= goals['goal_ball_precision_05'] else 0.0
        goal_achievements['goal_ball_map_05_achieved'] = 1.0 if ball_map_05 >= goals['goal_ball_map_05'] else 0.0
        goal_achievements['goal_ball_avg_predictions_achieved'] = 1.0 if ball_avg_preds >= goals['goal_ball_avg_predictions_per_image'] else 0.0
        
        # Log goal achievement metrics (with error handling and debugging)
        try:
            for goal_name, achieved in goal_achievements.items():
                if self.mlflow_debugger and self.use_mlflow:
                    context = self.mlflow_debugger.log_operation_start(
                        'log_metric',
                        metric_name=goal_name,
                        value=achieved,
                        step=epoch
                    )
                else:
                    context = None
                start_time = time.time()
                mlflow.log_metric(goal_name, achieved, step=epoch)
                if self.mlflow_debugger and self.use_mlflow:
                    self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
            
            # Log goal vs actual comparison (as percentage of goal)
            goal_progress = {
                'goal_player_recall_05_progress': (player_recall_05 / goals['goal_player_recall_05']) * 100 if goals['goal_player_recall_05'] > 0 else 0.0,
                'goal_player_precision_05_progress': (player_precision_05 / goals['goal_player_precision_05']) * 100 if goals['goal_player_precision_05'] > 0 else 0.0,
                'goal_player_map_05_progress': (player_map_05 / goals['goal_player_map_05']) * 100 if goals['goal_player_map_05'] > 0 else 0.0,
                'goal_player_map_75_progress': (player_map_75 / goals['goal_player_map_75']) * 100 if goals['goal_player_map_75'] > 0 else 0.0,
                'goal_ball_recall_05_progress': (ball_recall_05 / goals['goal_ball_recall_05']) * 100 if goals['goal_ball_recall_05'] > 0 else 0.0,
                'goal_ball_precision_05_progress': (ball_precision_05 / goals['goal_ball_precision_05']) * 100 if goals['goal_ball_precision_05'] > 0 else 0.0,
                'goal_ball_map_05_progress': (ball_map_05 / goals['goal_ball_map_05']) * 100 if goals['goal_ball_map_05'] > 0 else 0.0,
                'goal_ball_avg_predictions_progress': (ball_avg_preds / goals['goal_ball_avg_predictions_per_image']) * 100 if goals['goal_ball_avg_predictions_per_image'] > 0 else 0.0,
            }
            
            for progress_name, progress_value in goal_progress.items():
                if self.mlflow_debugger and self.use_mlflow:
                    context = self.mlflow_debugger.log_operation_start(
                        'log_metric',
                        metric_name=progress_name,
                        value=min(progress_value, 100.0),
                        step=epoch
                    )
                else:
                    context = None
                start_time = time.time()
                mlflow.log_metric(progress_name, min(progress_value, 100.0), step=epoch)  # Cap at 100%
                if self.mlflow_debugger and self.use_mlflow:
                    self.mlflow_debugger.log_operation_success('log_metric', time.time() - start_time, context)
        except Exception as mlflow_error:
            # Log detailed error information
            if self.mlflow_debugger and self.use_mlflow:
                self.mlflow_debugger.log_operation_error(
                    'log_metric',
                    mlflow_error,
                    context={'epoch': epoch, 'operation': 'goal_tracking'},
                    epoch=epoch
                )
            # Don't let MLflow errors stop goal tracking
            if MLFLOW_DEBUG:
                print(f"[MLFLOW DEBUG] Goal tracking MLflow logging failed (non-blocking): {type(mlflow_error).__name__}: {mlflow_error}")
            pass  # Silently ignore goal tracking MLflow errors
    
    def save_checkpoint(self, epoch: int, map_score: float, is_best: bool = False, 
                       is_interrupt: bool = False, lightweight: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            map_score: Validation mAP score (0.0 if not validated)
            is_best: Whether this is the best model so far
            is_interrupt: Whether this is an interrupt/error save
            lightweight: If True, save only model weights (faster, for frequent saves)
        """
        # Force lightweight for interrupt saves or if config requires it
        use_lightweight_only = self.config['checkpoint'].get('use_lightweight_only', False)
        if is_interrupt or use_lightweight_only:
            lightweight = True
        
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if lightweight:
            # Lightweight checkpoint: just model weights and epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_lightweight.pth"
        else:
            # Full checkpoint: includes optimizer, scheduler, metrics
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'map': map_score,
                'config': self.config
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        # Save checkpoint with error handling for disk quota issues
        try:
            torch.save(checkpoint, checkpoint_path)
        except (OSError, IOError) as e:
            if "Disk quota exceeded" in str(e) or "No space left" in str(e) or "file write failed" in str(e):
                print(f"ERROR: Disk quota exceeded. Failed to save checkpoint: {checkpoint_path}")
                print(f"Attempting to save lightweight checkpoint instead...")
                # Try lightweight version if full checkpoint failed
                if not lightweight:
                    lightweight_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'config': self.config
                    }
                    lightweight_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_lightweight.pth"
                    try:
                        torch.save(lightweight_checkpoint, lightweight_path)
                        print(f"Saved lightweight checkpoint instead: {lightweight_path}")
                        checkpoint_path = lightweight_path
                    except Exception as e2:
                        print(f"ERROR: Failed to save even lightweight checkpoint: {e2}")
                        raise
                else:
                    raise
            else:
                raise
        
        if is_interrupt:
            print(f"Saved interrupt checkpoint: {checkpoint_path}")
        elif lightweight:
            print(f"Saved lightweight checkpoint: {checkpoint_path}")
        else:
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model (only for full checkpoints with validation, and if enabled)
        if is_best and not lightweight and self.config['checkpoint'].get('save_best', True):
            best_path = checkpoint_dir / "best_model.pth"
            try:
                torch.save(checkpoint, best_path)
                print(f"Saved best model with mAP: {map_score:.4f}")
            except (OSError, IOError) as e:
                if "Disk quota exceeded" in str(e) or "No space left" in str(e) or "file write failed" in str(e):
                    print(f"WARNING: Disk quota exceeded. Skipping best model save.")
                else:
                    raise
        
        # Save latest checkpoint (always overwrite)
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        try:
            torch.save(checkpoint, latest_path)
        except (OSError, IOError) as e:
            if "Disk quota exceeded" in str(e) or "No space left" in str(e) or "file write failed" in str(e):
                print(f"WARNING: Disk quota exceeded. Skipping latest checkpoint save.")
            else:
                raise
        
        # Log checkpoint to MLflow (with retry logic and error recovery)
        if self.use_mlflow:
            # Circuit breaker: skip if too many failures
            if self.mlflow_failure_count >= self.mlflow_max_failures:
                if epoch % 10 == 0:  # Only warn occasionally
                    print(f"WARNING: MLflow disabled due to {self.mlflow_failure_count} consecutive failures")
                return
            
            max_retries = 2
            retry_count = 0
            logged_successfully = False
            
            while retry_count < max_retries and not logged_successfully:
                model_was_eval = False
                try:
                    if not lightweight:
                        # Pre-operation validation for checkpoint artifact
                        if self.mlflow_debugger and self.use_mlflow:
                            # Check filesystem health
                            fs_health = self.mlflow_debugger.check_filesystem_health(str(checkpoint_path.parent))
                            if fs_health['disk_percent'] > 90:
                                print(f"[MLFLOW DEBUG] WARNING: Disk usage is {fs_health['disk_percent']:.1f}% - may cause issues")
                            
                            # Validate checkpoint file
                            if not os.path.exists(checkpoint_path):
                                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
                            if not os.access(checkpoint_path, os.R_OK):
                                raise PermissionError(f"Checkpoint file is not readable: {checkpoint_path}")
                            
                            checkpoint_size = os.path.getsize(checkpoint_path)
                            if checkpoint_size == 0:
                                raise ValueError(f"Checkpoint file is empty: {checkpoint_path}")
                            
                            # Log checkpoint artifact
                            context = self.mlflow_debugger.log_operation_start(
                                'log_artifact',
                                file_path=str(checkpoint_path),
                                artifact_path="checkpoints",
                                file_size=checkpoint_size
                            )
                        else:
                            context = None
                        
                        start_time = time.time()
                        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                        
                        if self.mlflow_debugger and self.use_mlflow:
                            self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                            
                            # Post-operation verification
                            # Check if artifact was written (for file backend, check mlruns directory)
                            if self.mlflow_debugger.tracking_uri.startswith('file:'):
                                artifact_root = self.mlflow_debugger.tracking_uri.replace('file:', '')
                                if self.mlflow_debugger.run_id:
                                    artifact_path = os.path.join(artifact_root, self.mlflow_debugger.run_id, "artifacts", "checkpoints", checkpoint_path.name)
                                    if os.path.exists(artifact_path):
                                        written_size = os.path.getsize(artifact_path)
                                        print(f"[MLFLOW DEBUG] Verified artifact written: {artifact_path} ({self.mlflow_debugger._format_size(written_size)})")
                                    else:
                                        print(f"[MLFLOW DEBUG] WARNING: Artifact verification failed - file not found at {artifact_path}")
                        
                        if is_best and os.path.exists(best_path):
                            # Pre-operation validation for best model artifact
                            if self.mlflow_debugger and self.use_mlflow:
                                best_size = os.path.getsize(best_path)
                                context = self.mlflow_debugger.log_operation_start(
                                    'log_artifact',
                                    file_path=str(best_path),
                                    artifact_path="checkpoints",
                                    file_size=best_size
                                )
                            else:
                                context = None
                            
                            start_time = time.time()
                            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
                            
                            if self.mlflow_debugger and self.use_mlflow:
                                self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                    
                    # Save model in MLflow's native format every epoch (skip if model logging disabled)
                    if self.mlflow_model_logging_enabled:
                        # Set model to eval mode for inference
                        self.model.eval()
                        model_was_eval = True
                        model_log_retry = 0
                        model_logged = False
                        
                        while model_log_retry < max_retries and not model_logged:
                            temp_model_path = None
                            temp_dir = None
                            context = None
                            try:
                                # Pre-operation validation
                                if self.mlflow_debugger and self.use_mlflow:
                                    # Check backend health before model logging
                                    backend_health = self.mlflow_debugger.check_mlflow_backend_health()
                                    if not backend_health['backend_accessible']:
                                        raise RuntimeError("MLflow backend not accessible")
                                    if not backend_health['run_active']:
                                        raise RuntimeError("MLflow run is not active")
                                    
                                    # Check filesystem health
                                    fs_health = self.mlflow_debugger.check_filesystem_health()
                                    if fs_health['disk_percent'] > 90:
                                        print(f"[MLFLOW DEBUG] WARNING: Disk usage is {fs_health['disk_percent']:.1f}% - may cause issues")
                                
                                # FIX: Save model to temp file first, then log the file to avoid INTERNAL_ERROR
                                # This prevents issues with model state, device placement, or serialization
                                import tempfile
                                temp_dir = tempfile.mkdtemp(prefix=f"mlflow_model_epoch_{epoch}_")
                                temp_model_path = os.path.join(temp_dir, "model.pth")
                                
                                # Validate temp directory
                                if not os.path.exists(temp_dir) or not os.access(temp_dir, os.W_OK):
                                    raise OSError(f"Temp directory not writable: {temp_dir}")
                                
                                # Save model state dict (lighter and more reliable than full model)
                                # Use state_dict() which doesn't require moving model to CPU
                                if self.mlflow_debugger and self.use_mlflow:
                                    context = self.mlflow_debugger.log_operation_start(
                                        'save_model_temp',
                                        file_path=temp_model_path,
                                        operation="saving model state dict to temp file"
                                    )
                                
                                # Get state dict without moving model (more efficient)
                                model_state_dict = {k: v.cpu() if v.is_cuda else v for k, v in self.model.state_dict().items()}
                                
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model_state_dict,
                                    'map': map_score,
                                    'is_best': is_best,
                                    'config': self.config
                                }, temp_model_path)
                                
                                # Verify temp file was created
                                if not os.path.exists(temp_model_path):
                                    raise FileNotFoundError(f"Failed to create temp model file: {temp_model_path}")
                                
                                temp_file_size = os.path.getsize(temp_model_path)
                                if temp_file_size == 0:
                                    raise ValueError(f"Temp model file is empty: {temp_model_path}")
                                
                                if self.mlflow_debugger and self.use_mlflow:
                                    self.mlflow_debugger.log_operation_success('save_model_temp', time.time() - context['start_time'], context)
                                    print(f"[MLFLOW DEBUG] Model saved to temp file: {temp_model_path} ({self.mlflow_debugger._format_size(temp_file_size)})")
                                
                                # Use different artifact paths for different epochs (sanitize for MLflow)
                                model_artifact_path = f"models_epoch_{epoch}"
                                
                                # Validate artifact path (no invalid characters)
                                invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
                                if any(char in model_artifact_path for char in invalid_chars):
                                    raise ValueError(f"Invalid artifact path characters: {model_artifact_path}")
                                
                                # Pre-operation validation for artifact logging
                                if self.mlflow_debugger and self.use_mlflow:
                                    fs_health = self.mlflow_debugger.check_filesystem_health(temp_dir)
                                    if fs_health['disk_free'] < temp_file_size * 2:  # Need at least 2x file size free
                                        raise OSError(f"Insufficient disk space: {self.mlflow_debugger._format_size(fs_health['disk_free'])} free, need {self.mlflow_debugger._format_size(temp_file_size * 2)}")
                                    
                                    context = self.mlflow_debugger.log_operation_start(
                                        'log_artifact',
                                        file_path=temp_model_path,
                                        artifact_path=model_artifact_path,
                                        file_size=temp_file_size
                                    )
                                
                                # Log the saved model file instead of the live model object
                                start_time = time.time()
                                mlflow.log_artifact(temp_model_path, artifact_path=model_artifact_path)
                                
                                if self.mlflow_debugger and self.use_mlflow:
                                    self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                                    
                                    # Post-operation verification
                                    if self.mlflow_debugger.tracking_uri.startswith('file:'):
                                        artifact_root = self.mlflow_debugger.tracking_uri.replace('file:', '')
                                        if self.mlflow_debugger.run_id:
                                            artifact_path = os.path.join(artifact_root, self.mlflow_debugger.run_id, "artifacts", model_artifact_path, "model.pth")
                                            if os.path.exists(artifact_path):
                                                written_size = os.path.getsize(artifact_path)
                                                print(f"[MLFLOW DEBUG] Verified model artifact written: {artifact_path} ({self.mlflow_debugger._format_size(written_size)})")
                                            else:
                                                print(f"[MLFLOW DEBUG] WARNING: Model artifact verification failed - file not found at {artifact_path}")
                                
                                # Also log metadata as a separate artifact
                                metadata_path = os.path.join(temp_dir, "metadata.yaml")
                                import yaml
                                with open(metadata_path, 'w') as f:
                                    yaml.dump({
                                        "epoch": epoch,
                                        "map": map_score,
                                        "is_best": is_best,
                                        "model_config": self.config.get('model', {}),
                                        "training_config": self.config.get('training', {})
                                    }, f)
                                
                                # Log metadata artifact
                                if self.mlflow_debugger and self.use_mlflow:
                                    context = self.mlflow_debugger.log_operation_start(
                                        'log_artifact',
                                        file_path=metadata_path,
                                        artifact_path=model_artifact_path,
                                        file_size=os.path.getsize(metadata_path)
                                    )
                                start_time = time.time()
                                mlflow.log_artifact(metadata_path, artifact_path=model_artifact_path)
                                if self.mlflow_debugger and self.use_mlflow:
                                    self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                                
                                if is_best:
                                    # Also save best model at standard "model" path for easy access
                                    if self.mlflow_debugger and self.use_mlflow:
                                        context = self.mlflow_debugger.log_operation_start(
                                            'log_artifact',
                                            file_path=temp_model_path,
                                            artifact_path="model",
                                            file_size=temp_file_size
                                        )
                                    start_time = time.time()
                                    mlflow.log_artifact(temp_model_path, artifact_path="model")
                                    if self.mlflow_debugger and self.use_mlflow:
                                        self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                                    
                                    if self.mlflow_debugger and self.use_mlflow:
                                        context = self.mlflow_debugger.log_operation_start(
                                            'log_artifact',
                                            file_path=metadata_path,
                                            artifact_path="model",
                                            file_size=os.path.getsize(metadata_path)
                                        )
                                    start_time = time.time()
                                    mlflow.log_artifact(metadata_path, artifact_path="model")
                                    if self.mlflow_debugger and self.use_mlflow:
                                        self.mlflow_debugger.log_operation_success('log_artifact', time.time() - start_time, context)
                                    
                                    print(f"Saved model to MLflow - epoch {epoch} (mAP: {map_score:.4f}, BEST)")
                                else:
                                    print(f"Saved model to MLflow - epoch {epoch} (mAP: {map_score:.4f})")
                                
                                model_logged = True
                                
                                # Clean up temp files
                                import shutil
                                try:
                                    if temp_dir and os.path.exists(temp_dir):
                                        shutil.rmtree(temp_dir)
                                except Exception as cleanup_error:
                                    if MLFLOW_DEBUG:
                                        print(f"[MLFLOW DEBUG] Warning: Failed to cleanup temp dir {temp_dir}: {cleanup_error}")
                                    
                            except Exception as model_log_error:
                                model_log_retry += 1
                                
                                # Log detailed error information
                                if self.mlflow_debugger and self.use_mlflow:
                                    self.mlflow_debugger.log_operation_error(
                                        'log_artifact',
                                        model_log_error,
                                        context=context,
                                        retry_count=model_log_retry,
                                        max_retries=max_retries,
                                        epoch=epoch,
                                        file_path=temp_model_path if temp_model_path else None
                                    )
                                
                                error_type = type(model_log_error).__name__
                                error_msg = f"Failed to log model to MLflow (attempt {model_log_retry}/{max_retries}): {error_type}: {model_log_error}"
                                
                                # Clean up temp files on error
                                if temp_dir and os.path.exists(temp_dir):
                                    try:
                                        import shutil
                                        shutil.rmtree(temp_dir)
                                    except Exception as cleanup_error:
                                        if MLFLOW_DEBUG:
                                            print(f"[MLFLOW DEBUG] Warning: Failed to cleanup temp dir on error: {cleanup_error}")
                                
                                # Check if recoverable
                                recoverable = any(err in str(model_log_error) or err in error_type for err in ('RestException', 'ConnectionError', 'Timeout', 'INTERNAL_ERROR'))
                                
                                if model_log_retry < max_retries and recoverable:
                                    if MLFLOW_DEBUG:
                                        print(f"[MLFLOW DEBUG] {error_msg} - Retrying...")
                                    time.sleep(2)  # Longer delay for model logging
                                else:
                                    print(f"Warning: {error_msg}")
                                    # Don't disable model logging permanently - just skip this epoch
                                    # The temp file approach should fix INTERNAL_ERROR
                                    break
                        
                        # Restore training mode
                        if model_was_eval:
                            self.model.train()
                            model_was_eval = False
                    
                    logged_successfully = True
                    self.mlflow_failure_count = 0  # Reset failure count on success
                            
                except Exception as e:
                    retry_count += 1
                    
                    # Log detailed error information
                    if self.mlflow_debugger and self.use_mlflow:
                        self.mlflow_debugger.log_operation_error(
                            'log_checkpoint',
                            e,
                            context={'epoch': epoch, 'lightweight': lightweight},
                            retry_count=retry_count,
                            max_retries=max_retries,
                            checkpoint_path=str(checkpoint_path) if 'checkpoint_path' in locals() else None
                        )
                    
                    error_type = type(e).__name__
                    error_msg = f"Failed to log checkpoint to MLflow (attempt {retry_count}/{max_retries}): {error_type}: {e}"
                    
                    # Restore training mode if needed
                    if model_was_eval:
                        self.model.train()
                        model_was_eval = False
                    
                    # Increment failure count
                    self.mlflow_failure_count += 1
                    
                    # Check if recoverable
                    recoverable = any(err in str(e) or err in error_type for err in ('RestException', 'ConnectionError', 'Timeout', 'INTERNAL_ERROR'))
                    
                    if retry_count < max_retries and recoverable:
                        if MLFLOW_DEBUG:
                            print(f"[MLFLOW DEBUG] {error_msg} - Retrying...")
                        time.sleep(1)
                    else:
                        print(f"Warning: {error_msg} (failure count: {self.mlflow_failure_count}/{self.mlflow_max_failures})")
                        # Disable MLflow if too many failures
                        if self.mlflow_failure_count >= self.mlflow_max_failures:
                            print(f"WARNING: MLflow disabled after {self.mlflow_failure_count} consecutive failures. Training continues without MLflow.")
                            self.use_mlflow = False
                        break
    
    def train(self, start_epoch: int = 0, num_epochs: int = 50):
        """Main training loop"""
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        
        try:
            for epoch in range(start_epoch, num_epochs):
                if self.interrupted:
                    print("\nTraining interrupted. Saving checkpoint...")
                    # Save checkpoint without validation
                    self.save_checkpoint(epoch - 1, 0.0, is_best=False, is_interrupt=True)
                    break
                
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*50}")
                print(f"[VERBOSE] Starting epoch {epoch + 1} at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                
                # Train
                print(f"[VERBOSE] About to call train_epoch({epoch})", flush=True)
                train_loss = self.train_epoch(epoch)
                print(f"[VERBOSE] train_epoch({epoch}) completed, loss: {train_loss:.4f}", flush=True)
                print(f"Training Loss: {train_loss:.4f}")
                
                # Save checkpoint every epoch (for safety, even if not validating)
                # This ensures we don't lose progress if training stops early
                save_every_epoch = self.config['checkpoint'].get('save_every_epoch', True)
                if save_every_epoch:
                    # Save lightweight checkpoint every epoch
                    self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=False, lightweight=True)
                    
                    # Keep only last N lightweight checkpoints to save space (keep last 20)
                    # This balances frequent saves with disk space
                    keep_last_n = self.config['checkpoint'].get('keep_last_lightweight', 20)
                    if epoch >= keep_last_n:
                        import os
                        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
                        old_lightweight = checkpoint_dir / f"checkpoint_epoch_{epoch - keep_last_n}_lightweight.pth"
                        if old_lightweight.exists():
                            try:
                                old_lightweight.unlink()
                            except:
                                pass  # Ignore deletion errors
                
                # Validate less frequently for speed (every 10 epochs instead of 5)
                validate_frequency = 10
                if (epoch + 1) % validate_frequency == 0 or epoch == num_epochs - 1:
                    map_score = self.validate(epoch)
                    
                    # Save full checkpoint with validation metrics
                    is_best = map_score > self.best_map
                    if is_best:
                        self.best_map = map_score
                    
                    # Only save full checkpoint if not using lightweight-only mode
                    use_lightweight_only = self.config['checkpoint'].get('use_lightweight_only', False)
                    if not use_lightweight_only and (epoch + 1) % self.config['checkpoint']['save_frequency'] == 0:
                        self.save_checkpoint(epoch, map_score, is_best, is_interrupt=False, lightweight=False)
                
                # Print adaptive optimization stats at end of epoch
                if self.adaptive_optimizer:
                    stats = self.adaptive_optimizer.get_statistics()
                    print(f"\n📊 Adaptive Optimization Stats:")
                    print(f"   Adjustments made: {stats['adjustment_count']}")
                    print(f"   Current workers: {stats['current_workers']}, prefetch: {stats['current_prefetch']}")
                    print(f"   Avg GPU util: {stats['avg_gpu_utilization']:.1%}, Avg RAM: {stats['avg_ram_usage']:.1%}")
                
                print(f"{'='*50}\n")
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=True)
            raise
        
        except Exception as e:
            print(f"\n\nTraining error: {e}")
            print("Saving checkpoint before exit...")
            self.save_checkpoint(epoch, 0.0, is_best=False, is_interrupt=True)
            raise
