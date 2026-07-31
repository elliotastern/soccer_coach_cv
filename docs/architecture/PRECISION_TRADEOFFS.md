# Precision & Quantization Strategy

## Official Strategy (Per Requirements)

### 1. **Training: Mixed Precision (FP16/FP32)**
- **Status:** ✅ **ACTIVE** (RF-DETR uses `amp=True` by default)
- **Rationale:** Essential to capture tiny gradients of small objects (<15 pixels)
- **Implementation:** Automatic Mixed Precision (AMP) via PyTorch
- **Result:** ~2x faster training with minimal accuracy loss

### 2. **MVP Deployment: FP16 (Half Precision)**
- **Status:** ✅ **ACTIVE** for CUDA, ⚠️ **NEEDS UPDATE** for CPU
- **Rationale:** Safest start. ~3x speedup on NVIDIA GPUs with zero accuracy loss
- **Current:** CUDA uses FP16 ✅, CPU uses INT8 (PTQ) ⚠️
- **Action:** Use FP16 for MVP deployment (both CUDA and CPU)

### 3. **Future Optimization: INT8 via QAT (Quantization-Aware Training)**
- **Status:** 🔄 **FUTURE** (Only if FP16 is too slow)
- **Rationale:** Do NOT use PTQ (Post-Training Quantization). Re-train with QAT to ensure ball detection at 8-bit precision
- **When:** Edge devices, very slow inference requirements
- **Important:** Must use QAT, not PTQ, to preserve tiny object detection

---

## Available Options

### **Training Phase**

#### 1. **FP32 (Full Precision)** - `amp=False` or `mixed_precision: false`
**What it is:**
- All operations use 32-bit floating point
- Highest numerical precision

**Pros:**
- ✅ Highest accuracy - no precision loss
- ✅ Most stable training (no gradient underflow/overflow)
- ✅ Best for tiny objects (<15 pixels) - preserves fine details
- ✅ No risk of training instability
- ✅ Reproducible results

**Cons:**
- ❌ ~2x slower training speed
- ❌ ~2x higher GPU memory usage
- ❌ Can't fit as large batch sizes
- ❌ Longer training time

**Best for:**
- Tiny object detection (like your ball <15 pixels)
- When accuracy is critical
- When you have GPU memory to spare
- Debugging training issues

---

#### 2. **Mixed Precision (AMP - FP16/FP32)** - `amp=True` (Current Default)
**What it is:**
- Automatic Mixed Precision: FP16 for speed, FP32 for stability
- PyTorch automatically chooses which ops use FP16 vs FP32
- Uses gradient scaling to prevent underflow

**Pros:**
- ✅ ~1.5-2x faster training than FP32
- ✅ ~50% lower GPU memory usage
- ✅ Can fit larger batch sizes
- ✅ Minimal accuracy loss (usually <1%)
- ✅ Industry standard for training

**Cons:**
- ⚠️ Small accuracy loss possible (usually negligible)
- ⚠️ Slight risk of gradient underflow (rare, handled by scaler)
- ⚠️ May affect very tiny objects slightly

**Best for:**
- Most training scenarios (current default)
- When you need faster training
- When GPU memory is limited
- Production training pipelines

---

### **Inference Phase**

#### 3. **FP32 (Full Precision)** - No quantization
**What it is:**
- Full 32-bit precision during inference

**Pros:**
- ✅ Highest accuracy
- ✅ No quantization artifacts
- ✅ Best for tiny objects

**Cons:**
- ❌ Slowest inference (~2x slower than FP16)
- ❌ Highest memory usage
- ❌ Not suitable for real-time applications

**Best for:**
- Offline processing
- When accuracy is critical
- CPU inference (no GPU)

---

#### 4. **FP16 (Half Precision)** - `model.half()` (Current for CUDA)
**What it is:**
- All operations use 16-bit floating point
- Direct conversion from FP32 model

**Pros:**
- ✅ ~2x faster inference than FP32
- ✅ ~50% lower memory usage
- ✅ Minimal accuracy loss (<1% typically)
- ✅ Works on modern GPUs (Tensor Cores)

**Cons:**
- ⚠️ Small accuracy loss
- ⚠️ May affect very small objects slightly
- ❌ Not supported on older GPUs

**Best for:**
- Real-time inference on GPU
- When you need speed + accuracy balance
- Modern GPUs (V100, A100, RTX series)

---

#### 5. **INT8 Dynamic Quantization** - `quantize_dynamic()` (Current for CPU)
**What it is:**
- 8-bit integer quantization
- Weights stored as INT8, activations computed in INT8
- Dynamic: quantization scale computed at runtime

**Pros:**
- ✅ ~4x faster inference than FP32
- ✅ ~75% lower memory usage
- ✅ Best for CPU inference
- ✅ Can run on edge devices

**Cons:**
- ⚠️ Larger accuracy loss (2-5% typical)
- ⚠️ May significantly affect tiny object detection
- ⚠️ Not ideal for <15 pixel objects
- ❌ More complex deployment

**Best for:**
- CPU inference
- Edge devices / mobile
- When speed > accuracy
- Large batch inference

---

## Recommendations for Your Ball Detection Task

### **Training:**
**Current: Mixed Precision (AMP)** ✅ **RECOMMENDED**
- Your ball is tiny (<15 pixels), but AMP loss is usually <1%
- 2x faster training is worth the minimal accuracy tradeoff
- You can always fine-tune with FP32 if needed

**Alternative: FP32** (if you have issues)
- Only if you see training instability or accuracy problems
- Will be slower but more stable

### **Inference:**
**Current Setup:**
- **CUDA:** FP16 ✅ Good balance
- **CPU:** INT8 ⚠️ **Consider FP32 for tiny balls**

**Recommendation:**
- **GPU (CUDA):** Keep FP16 - good speed/accuracy balance
- **CPU:** Consider FP32 instead of INT8 for tiny ball detection
  - INT8 may lose small ball detections
  - FP32 on CPU is acceptable for offline processing

---

## Performance Comparison (Estimated)

| Option | Training Speed | Inference Speed | Memory | Accuracy Loss | Best For |
|--------|---------------|----------------|--------|---------------|----------|
| **FP32 Training** | 1.0x (baseline) | 1.0x | 100% | 0% | Tiny objects, debugging |
| **AMP Training** | 1.5-2.0x | - | 50% | <1% | **Recommended** |
| **FP32 Inference** | - | 1.0x | 100% | 0% | Offline, accuracy-critical |
| **FP16 Inference** | - | 2.0x | 50% | <1% | **GPU real-time** |
| **INT8 Inference** | - | 4.0x | 25% | 2-5% | CPU, edge devices |

---

## How to Change Settings

### Training (RF-DETR):
Currently RF-DETR uses `amp=True` by default. To disable:
- You'd need to modify RF-DETR's internal args (not easily configurable)
- Or use custom DETR trainer with `mixed_precision: false` in config

### Inference:
Modify `src/perception/local_detector.py`:
```python
# For FP32 (full precision):
# Comment out quantization code, use model as-is

# For FP16 (current CUDA):
self.model = self.model.half()  # Current

# For INT8 (current CPU):
self.model = torch.quantization.quantize_dynamic(...)  # Current
```

---

## Bottom Line

**For your tiny ball detection (<15 pixels):**
1. **Training:** Keep AMP (mixed precision) - minimal loss, 2x speed
2. **Inference GPU:** Keep FP16 - good balance
3. **Inference CPU:** Consider FP32 instead of INT8 - INT8 may lose tiny balls
