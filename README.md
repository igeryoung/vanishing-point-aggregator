# Vanishing Point Aggregator (VPA)

A PyTorch implementation of **VPA**, a novel approach for 3D Semantic Scene Completion (SSC) that leverages vanishing point aggregation to enhance multi-scale feature fusion for autonomous driving applications.

## Overview

This project implements an advanced 3D semantic scene completion system that uses vanishing point information to aggregate multi-scale features more effectively. The system combines:

- **Vanishing Point Aggregation**: Novel use of vanishing points to guide feature aggregation from multiple scales
- **Deformable Attention Mechanisms**: 3D deformable attention for efficient spatial reasoning
- **Multi-Scale Feature Fusion**: Hierarchical feature processing at different resolution levels
- **Region-Aware Processing**: Specialized attention modules for handling different spatial regions

## Key Features

- ✅ **Multi-dataset Support**: SemanticKITTI, KITTI-360
- ✅ **Flexible Architecture**: Modular design with configurable encoders and decoders
- ✅ **Lightning Integration**: Built on PyTorch Lightning for scalable training
- ✅ **Hydra Configuration**: Flexible configuration management

## Architecture

The system consists of several key components:

### Core Components
- **vpa Model** (`ssc_pl/models/segmentors/vpa.py`): Main architecture combining encoder-decoder with vanishing point processing
- **Voxel Proposal Layer** (`ssc_pl/models/projections/vpl.py`): Projects 2D features to 3D voxel space using deformable attention
- **Region Attention Module** (`ssc_pl/models/layers/fuse_vp_region.py`): Processes vanishing point regions with specialized attention
- **Deformable 3D Attention** (`ssc_pl/models/layers/deform_attn_3d.py`): Efficient 3D spatial attention mechanism

### Supported Encoders
- MMDetection wrapper for various 2D detection backbones
- UNet2D for feature extraction
- Custom encoder architectures

### Projection Methods
- **CVT**: Cross-View Transformer projection
- **FLOSP**: Feature Line-of-Sight Projection  
- **VPL**: Voxel Proposal Layer (recommended)

## Installation

### Requirements
```bash
torch
torchvision
lightning
hydra-core
numba
scikit-image
einops
yacs
```

### Setup
1. Clone the repository:
```bash
git clone &lt;repository-url&gt;
cd vanishing-point-aggregator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the Python path:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Dataset Preparation

### SemanticKITTI
Download the SemanticKITTI dataset and organize it as follows:
```
/path/to/SemanticKITTI/
├── dataset/
│   ├── sequences/
│   ├── labels/
│   └── depth/
```

Update the dataset configuration in `configs/datasets/semantic_kitti.yaml`:
```yaml
data:
  datasets:
    data_root: /path/to/SemanticKITTI
    label_root: /path/to/SemanticKITTI/dataset/labels
    depth_root: /path/to/SemanticKITTI/dataset/depth
```

### Other Datasets
Similar configurations are available for:
- **NYU Depth V2**: `configs/datasets/nyu_v2.yaml`
- **KITTI-360**: `configs/datasets/kitti_360.yaml`

## Usage

### Training

```bash
python tools/train.py [--config-name config[.yaml]] [trainer.devices=4] \
    [+data_root=$DATA_ROOT] [+label_root=$LABEL_ROOT] [+depth_root=$DEPTH_ROOT]
```

**Configuration Options:**
- `--config-name`: Override the default config file
- `trainer.devices`: Number of GPUs to use
- Command-line overrides supported via Hydra syntax

**Examples:**
```bash
# Train with default settings
python tools/train.py

# Train with multiple GPUs
python tools/train.py trainer.devices=4

# Train with custom data paths
python tools/train.py +data_root=/custom/path +label_root=/custom/labels

# Use different model configuration
python tools/train.py --config-name config_custom models=monoscene
```

For more configuration options, refer to:
- [Hydra Configuration Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/)
- [Hydra Override Grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)

### Testing

Generate outputs for evaluation server submission:

```bash
python tools/test.py [+ckpt_path=path/to/checkpoint.ckpt]
```

### Visualization

1. **Generate outputs:**
```bash
python tools/generate_outputs.py [+ckpt_path=path/to/checkpoint.ckpt]
```

2. **Visualize results:**
```bash
python tools/visualize.py [+path=path/to/outputs]
```

### Additional Tools

- **FLOPS Counting**: `python tools/count_flops.py`
- **Reference Points Visualization**: `python tools/ref_points_vis.py`
- **Depth Rendering**: `python tools/render_depth.py`

## Configuration

The project uses Hydra for configuration management. Key configuration files:

### Main Config (`configs/config.yaml`)
```yaml
defaults:
  - datasets: semantic_kitti    # Dataset configuration
  - models: vpa         # Model architecture
  - schedules: adamw_lr2e-4_30e # Training schedule
```

### Model Configurations
- `configs/models/vpa.yaml`: Main vpa architecture
- `configs/models/monoscene.yaml`: Alternative MonoScene baseline

### Training Schedules
- `configs/schedules/adamw_lr1e-4_30e.yaml`: Conservative learning rate
- `configs/schedules/adamw_lr2e-4_30e.yaml`: Standard learning rate

## Model Architecture Details

### Vanishing Point Processing
The system identifies vanishing points in input images and uses them to:
1. Guide attention mechanisms towards important spatial regions
2. Aggregate multi-scale features more effectively
3. Improve geometric understanding of the scene

### Multi-Scale Feature Fusion
Features are processed at multiple scales:
- **Scale 1**: Full resolution features
- **Scale 2**: Half resolution (default volume scale)
- **Scale 4**: Quarter resolution for global context

### 3D Scene Representation
- **Voxel Grid**: 3D scene discretized into voxels (default: 256×256×32)
- **Semantic Classes**: 20 classes for SemanticKITTI
- **Voxel Size**: 0.2m per voxel (configurable)

## Results and Evaluation

The model is evaluated using standard SSC metrics:
- **IoU**: Intersection over Union for geometric completion
- **mIoU**: Mean IoU across semantic classes
- **SSC**: Combined geometric and semantic completion accuracy

Evaluation configurations are in `configs/datasets/` for each supported dataset.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on top of Symphonies, Voxformer, mmdetection
- Incorporates ideas from MonoScene, CVT, and deformable attention mechanisms
- SemanticKITTI dataset providers
- MMDetection framework for 2D detection components