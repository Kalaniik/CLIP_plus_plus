# CLIP++

A PyTorch implementation of CLIP with enhanced prompt learning capabilities.

## Features

- **Enhanced Prompt Learning**: Support for both text and visual prompts
- **Flexible Architecture**: Modular design for easy customization
- **Multiple Model Support**: Support for various CLIP model variants (ViT, ResNet)
- **Easy Integration**: Simple API for integration with existing projects

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from clip_plus_plus import ClipPlusPlus
from clip_plus_plus.utils import load_clip

# Load CLIP model
clip_model = load_clip(cfg)

# Create CLIP++ model
model = ClipPlusPlus(cfg, clip_model)

# Use the model
output = model(input_image, use_vp=True, prompt_weight=0.1)
```

## Usage

### Basic Usage

```python
import torch
from clip_plus_plus import ClipPlusPlus
from clip_plus_plus.utils import load_clip

# Configuration
class Config:
    model_name = "ViT-B/32"
    gpu_id = 0
    vp = True
    tp = True
    batch_size = 32
    ctx_init = "a photo of a"

cfg = Config()

# Load model
clip_model = load_clip(cfg)
model = ClipPlusPlus(cfg, clip_model)

# Forward pass
image = torch.randn(1, 3, 224, 224)
output = model(image, use_vp=True, prompt_weight=0.1)
```

### Custom Prompt Learning

```python
# Get trainable parameters
visual_params = model.get_visual_prompt_params()
all_params = model.get_trainable_params(update_tp=True, update_vp=True)

# Create optimizer
optimizer = torch.optim.Adam(all_params, lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## Model Architecture

CLIP++ extends the original CLIP architecture with:

1. **Visual Prompt Learning**: Learnable visual prompts that can be added to input images
2. **Text Prompt Learning**: Learnable text prompts for better text representation
3. **Flexible Integration**: Easy integration of both prompt types

## License

Apache-2.0 License

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{clip_plus_plus,
  title={CLIP++: Enhanced CLIP with Prompt Learning},
  author={Guo Kai},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Kalaniik/CLIP_plus_plus}
}
``` 
