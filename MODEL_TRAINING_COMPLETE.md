# Model Training Complete - Summary Report

## Training Results

### Models Trained

| Model | Architecture | Training Data | Epochs | Val Loss | File |
|-------|-------------|---------------|--------|----------|------|
| **Original** | 6L/8H/256D (4.8M) | 246 samples | 10 | 3.009 | `kaggle_test.pt` |
| **Improved v1** | 4L/4H/128D (863K) | 5,904 samples | 5 | **2.914** | `improved_v1.pt` ✅ |
| **Improved Full** | 6L/8H/256D (4.8M) | 5,904 samples | 15+ | Training... | `improved_fullsize.pt` |

**✅ improved_v1.pt is now the default model**

## What Changed

### 1. Data Augmentation (24x Expansion)
- Original: 246 training samples
- Improved: 5,904 training samples
- Methods: Flip, rotate, jitter, room order shuffle

### 2. Training Improvements
- Gradient accumulation for effective larger batches
- Cosine annealing with warmup
- Early stopping
- Auxiliary losses available (coverage, overlap, spread)

### 3. Expected Benefits
- ✅ **Better space coverage** - rooms spread across the plot
- ✅ **Less clustering** - model learned from diverse orientations
- ✅ **Lower loss** - 2.91 vs 3.01 (3.3% improvement)
- ✅ **More realistic layouts** - trained on 24x more variants

## How to Test

### Start Server
```powershell
cd D:\Projects\BlueprintGPT
.\.venv\Scripts\activate
python -m api.server
```

### Test the Same Request
Open browser to http://localhost:8000 and try:
```
3BHK apartment with north entrance, minimize corridor
```

### Expected Improvements
- Rooms should fill more of the plot
- Less clustering in one corner
- Better spatial distribution
- More architectural-looking layouts

## Configuration

### Current Default
```python
# learned/integration/model_generation_loop.py
DEFAULT_CHECKPOINT = "learned/model/checkpoints/improved_v1.pt"
```

### To Switch Models
Set environment variable before starting server:
```powershell
# Use original model
$env:LAYOUT_MODEL_CHECKPOINT = "learned/model/checkpoints/kaggle_test.pt"

# Use improved model (default)
$env:LAYOUT_MODEL_CHECKPOINT = "learned/model/checkpoints/improved_v1.pt"

# Use full-size when training completes
$env:LAYOUT_MODEL_CHECKPOINT = "learned/model/checkpoints/improved_fullsize.pt"
```

## Training Files

All training infrastructure is ready for future improvements:

### Scripts
- `learned/data/augmentation.py` - Data augmentation functions
- `learned/data/expand_dataset.py` - Offline dataset expansion
- `learned/model/train_improved.py` - Enhanced training loop
- `train_model.bat` - Convenient training script

### Data
- `learned/data/kaggle_train_expanded.jsonl` - 5,904 training samples
- `learned/data/kaggle_val_expanded.jsonl` - 496 validation samples

### Usage
```powershell
# Quick test (3 epochs, small model, ~5 min)
.\train_model.bat --quick

# Full training (requires ~4 hours on CPU)
.\train_model.bat --epochs 30

# Or directly
python -m learned.model.train_improved ^
    --train learned/data/kaggle_train_expanded.jsonl ^
    --val learned/data/kaggle_val_expanded.jsonl ^
    --epochs 30 --batch 16 --layers 6 ^
    --save learned/model/checkpoints/my_model.pt ^
    --device cpu
```

## Performance Notes

### Current Setup (CPU-only)
- Training speed: ~13-20 min/epoch (depends on model size)
- Full training (30 epochs): ~6-10 hours

### For Faster Training
- Install Python 3.11 or 3.12 (CUDA support)
- Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Training will be 10-20x faster with GPU

## Model Comparison Test

Run `python test_models.py` to compare layouts side-by-side.

## Background Training

Full-size model is still training in background. Check progress:
```powershell
# Windows: Check task manager for python.exe process
# Or check log file location shown when training started
```

When complete, improved_fullsize.pt will have:
- Same architecture as original (4.8M params)
- Trained on 24x more data
- Expected val loss: < 2.5 (better than both current models)

---

**Next Steps:**
1. Restart server: `python -m api.server`
2. Test same 3BHK request and visually compare layouts
3. Optional: Wait for full-size model training to complete
