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

## Acknowledgements
This project builds upon several foundational works in the field of prompt learning and visual-language pretraining. I sincerely thank the authors of the following methods for open-sourcing their code, which greatly facilitated our development.

If you use this repository, i encourage you to also cite the original papers and codebases listed below:

| Method       | Paper                                                   | Venue      | Code Link |
| ------------ | ------------------------------------------------------- | ---------- | --------- |
| CoOp         | [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)                                 | IJCV  2022  |  [link](https://github.com/KaiyangZhou/CoOp)  |
| CoCoOp       | [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)                        | CVPR  2022  |  [link](https://github.com/KaiyangZhou/CoOp)  |
| VPT          | [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274)                      | ECCV  2022  |  [link](https://github.com/KMnP/vpt)  |
| MaPLe        | [MaPLe: Multi-modal Prompt Learning](https://arxiv.org/abs/2210.03117)                                            | CVPR  2023  |  [link](https://github.com/muzairkhattak/multimodal-prompt-learning)  |
| DAPL         | [Domain Adaptation via Prompt Learning](https://arxiv.org/abs/2202.06687)                                         | TNNLS 2023  |  [link](https://github.com/LeapLabTHU/DAPrompt)  |
| PDA          | [Prompt-based Distribution Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/2312.09553)        | AAAI  2024  |  [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment)  |


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
