"""
Adaptive Resource Optimizer for Training
Dynamically adjusts training parameters based on real-time usage metrics
"""
import time
import torch
from typing import Dict, Optional
from collections import deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AdaptiveOptimizer:
    """Dynamically optimizes training parameters based on resource usage"""
    
    def __init__(self, 
                 initial_num_workers: int = 6,
                 initial_prefetch_factor: int = 2,
                 target_gpu_utilization: float = 0.85,
                 max_ram_usage: float = 0.80,
                 adjustment_interval: int = 50,
                 min_workers: int = 2,
                 max_workers: int = 16,
                 min_prefetch: int = 1,
                 max_prefetch: int = 4):
        """
        Initialize adaptive optimizer
        
        Args:
            initial_num_workers: Starting number of DataLoader workers
            initial_prefetch_factor: Starting prefetch factor
            target_gpu_utilization: Target GPU utilization (0.0-1.0)
            max_ram_usage: Maximum RAM usage threshold (0.0-1.0)
            adjustment_interval: How often to check and adjust (in batches)
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            min_prefetch: Minimum prefetch factor
            max_prefetch: Maximum prefetch factor
        """
        self.num_workers = initial_num_workers
        self.prefetch_factor = initial_prefetch_factor
        
        self.target_gpu_utilization = target_gpu_utilization
        self.max_ram_usage = max_ram_usage
        self.adjustment_interval = adjustment_interval
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_prefetch = min_prefetch
        self.max_prefetch = max_prefetch
        
        # Metrics tracking
        self.gpu_utilization_history = deque(maxlen=20)
        self.ram_usage_history = deque(maxlen=20)
        self.data_loading_times = deque(maxlen=20)
        self.gpu_processing_times = deque(maxlen=20)
        
        self.last_adjustment_batch = 0
        self.adjustment_count = 0
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {}
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics['gpu_utilization'] = self._get_gpu_utilization()
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            metrics['gpu_memory_percent'] = metrics['gpu_memory_used'] / metrics['gpu_memory_total']
        
        # RAM metrics
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            ram_info = process.memory_info()
            metrics['ram_used_gb'] = ram_info.rss / (1024 ** 3)
            # Get total system RAM
            total_ram = psutil.virtual_memory().total / (1024 ** 3)
            metrics['ram_total_gb'] = total_ram
            metrics['ram_percent'] = metrics['ram_used_gb'] / total_ram
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization (0.0-1.0)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 100.0
        except Exception:
            pass
        return 0.0
    
    def record_batch_timing(self, data_loading_time: float, gpu_processing_time: float):
        """Record timing metrics for a batch"""
        self.data_loading_times.append(data_loading_time)
        self.gpu_processing_times.append(gpu_processing_time)
    
    def should_adjust(self, batch_idx: int) -> bool:
        """Check if we should attempt adjustment"""
        if batch_idx - self.last_adjustment_batch < self.adjustment_interval:
            return False
        if len(self.gpu_utilization_history) < 10:
            return False
        return True
    
    def adjust_parameters(self, batch_idx: int) -> Optional[Dict[str, int]]:
        """
        Adjust parameters based on current metrics
        
        Returns:
            Dictionary with new parameters if adjustment was made, None otherwise
        """
        if not self.should_adjust(batch_idx):
            return None
        
        metrics = self.get_current_metrics()
        
        # Update history
        if 'gpu_utilization' in metrics:
            self.gpu_utilization_history.append(metrics['gpu_utilization'])
        if 'ram_percent' in metrics:
            self.ram_usage_history.append(metrics['ram_percent'])
        
        # Calculate averages
        avg_gpu_util = sum(self.gpu_utilization_history) / len(self.gpu_utilization_history) if self.gpu_utilization_history else 0.0
        avg_ram_usage = sum(self.ram_usage_history) / len(self.ram_usage_history) if self.ram_usage_history else 0.0
        
        # Calculate data loading vs processing ratio
        avg_data_time = sum(self.data_loading_times) / len(self.data_loading_times) if self.data_loading_times else 0.0
        avg_gpu_time = sum(self.gpu_processing_times) / len(self.gpu_processing_times) if self.gpu_processing_times else 0.0
        
        adjustments = {}
        adjustment_made = False
        
        # Strategy 1: If GPU utilization is low and RAM is available, increase data loading
        if avg_gpu_util < self.target_gpu_utilization * 0.7 and avg_ram_usage < self.max_ram_usage * 0.7:
            # GPU is underutilized and we have RAM headroom
            if self.num_workers < self.max_workers:
                self.num_workers = min(self.num_workers + 1, self.max_workers)
                adjustments['num_workers'] = self.num_workers
                adjustment_made = True
            
            if self.prefetch_factor < self.max_prefetch:
                self.prefetch_factor = min(self.prefetch_factor + 1, self.max_prefetch)
                adjustments['prefetch_factor'] = self.prefetch_factor
                adjustment_made = True
        
        # Strategy 2: If data loading is bottleneck (data_time > gpu_time), increase prefetch
        elif avg_data_time > 0 and avg_gpu_time > 0 and avg_data_time > avg_gpu_time * 1.2:
            if self.prefetch_factor < self.max_prefetch and avg_ram_usage < self.max_ram_usage * 0.8:
                self.prefetch_factor = min(self.prefetch_factor + 1, self.max_prefetch)
                adjustments['prefetch_factor'] = self.prefetch_factor
                adjustment_made = True
        
        # Strategy 3: If RAM is high, reduce workers/prefetch
        elif avg_ram_usage > self.max_ram_usage:
            if self.num_workers > self.min_workers:
                self.num_workers = max(self.num_workers - 1, self.min_workers)
                adjustments['num_workers'] = self.num_workers
                adjustment_made = True
            
            if self.prefetch_factor > self.min_prefetch:
                self.prefetch_factor = max(self.prefetch_factor - 1, self.min_prefetch)
                adjustments['prefetch_factor'] = self.prefetch_factor
                adjustment_made = True
        
        # Strategy 4: If GPU utilization is very high (>95%), we might be data-limited
        elif avg_gpu_util > 0.95 and avg_ram_usage < self.max_ram_usage * 0.8:
            if self.prefetch_factor < self.max_prefetch:
                self.prefetch_factor = min(self.prefetch_factor + 1, self.max_prefetch)
                adjustments['prefetch_factor'] = self.prefetch_factor
                adjustment_made = True
        
        if adjustment_made:
            self.last_adjustment_batch = batch_idx
            self.adjustment_count += 1
            adjustments['metrics'] = {
                'avg_gpu_utilization': avg_gpu_util,
                'avg_ram_usage': avg_ram_usage,
                'avg_data_loading_time': avg_data_time,
                'avg_gpu_processing_time': avg_gpu_time
            }
            return adjustments
        
        return None
    
    def get_current_settings(self) -> Dict[str, int]:
        """Get current parameter settings"""
        return {
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor
        }
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics"""
        return {
            'adjustment_count': self.adjustment_count,
            'current_workers': self.num_workers,
            'current_prefetch': self.prefetch_factor,
            'avg_gpu_utilization': sum(self.gpu_utilization_history) / len(self.gpu_utilization_history) if self.gpu_utilization_history else 0.0,
            'avg_ram_usage': sum(self.ram_usage_history) / len(self.ram_usage_history) if self.ram_usage_history else 0.0,
        }
