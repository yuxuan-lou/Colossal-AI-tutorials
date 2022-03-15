# From data parallel to hybrid parallel: Accelerate ViT training with Colossal-AI (step by step tutorial, multi-gpu)
[Code](https://github.com/yuxuan-lou/Step-by-Step-ViT-training-on-Cifar10-with-Colossal-AI)

Colossal-AI provides three different parallelism techniques which acclerate model training: data parallelism, pipeline parallelism and tensor parallelism. 
In this example, we will show you how to train ViT on Cifar10 dataset with these parallelism techniques. To run this example, you will need 2-4 GPUs. 

## Colossal-AI Installation
You can install Colossal-AI pacakage and its dependencies with PyPI.
```bash
pip install colossalai
```

## Access Example Code
```bash
git clone https://github.com/yuxuan-lou/Step-by-Step-ViT-training-on-Cifar10-with-Colossal-AI.git
```

## Data Parallelism
Data parallism is one basic way to accelerate model training process. You can apply data parallism to training by only two steps:
1. Define a configuration file
2. Change a few lines of code in train script

### Define your configuration file `config_data_parallel.py`
To use Colossal-AI, the first step is to define a configuration file. And there are two kinds of variables here:

1. **Colossal-AI feature specification**

There is an array of features Colossal-AI provides to speed up training (parallel mode, mixed precision, ZeRO, etc.). Each feature is defined by a corresponding field in the config file. If we apply data parallel only, we do not need to specify the parallel mode. In this example, we use mixed precision training natively provided by PyTorch by define the mixed precision configuration `fp16 = dict(mode=AMP_TYPE.TORCH)`.

2. **Global hyper-parameters**

Global hyper-parameters include model-specific hyper-parameters, training settings, dataset information, etc.

```python
from colossalai.amp import AMP_TYPE

# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 300

# mix precision
fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
clip_grad_norm = 1.0

dali = dict(
    gpu_aug=True,
    mixup_alpha=0.2
)
```



### Start training
`DATA` is the filepath where Cifar10 dataset will be automatically downloaded and stored.

`<NUM_GPUs>` is the number of GPUs you want to use to train ViT on Cifar10 with data parallelism.

```bash
export DATA=<path_to_data>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_dp.py --config ./configs/config_data_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_dp.py --config ./configs/config_data_parallel.py
# Otherwise
# python -m torch.distributed.launch --nproc_per_node <NUM_GPUs> --master_addr <node_name> --master_port 29500 train_dp.py --config ./configs/config.py
```

During Training:
<p align="center">
  <img src="https://github.com/yuxuan-lou/Colossal-AI-tutorials/blob/main/img/vit_dp.png" width="800">
</p>

## Pipeline Parallelism
Aside from data parallelism, Colossal-AI also support pipleline parallelism. In specific, Colossal-AI uses 1F1B pipeline introduced by Nvidia. For more details, you can view the related [documents](https://www.colossalai.org/tutorials/features/pipeline_parallel).

### Define your configuration file(`config_pipeline_parallel.py`)
To apply pipleline parallel on the data parallel basis, you only need to add a **parallel dict**
```python
from colossalai.amp import AMP_TYPE

parallel = dict(
    pipeline=2
)
# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0
```

Other configs：
```python
# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token
```

### Start training
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_pipeline_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_pipeline_parallel.py
```

Initial parallel settings and build pipeline model:
<p align="center">
  <img src="https://github.com/yuxuan-lou/Colossal-AI-tutorials/blob/main/img//vit_pp_1.png" width="800">
</p>

During training:
<p align="center">
  <img src="https://github.com/yuxuan-lou/Colossal-AI-tutorials/blob/main/img/vit_pp_2.png" width="800">
</p>


## Tensor Parallelism and Hybrid Parallelism
Tensor parallelism partitions each weight parameter across multiple devices in order to reduce memory load. Colossal-AI support 1D, 2D, 2.5D and 3D tensor parallelism. Besides, you can combine tensor parallelism with pipeline parallelism and data parallelism to reach hybrid parallelism. Colossal-AI also provides an easy way to apply tensor parallelism and hybrid parallelism. A few lines of code changing is all you need.

### Define your configuration file(`config_hybrid_parallel.py`)
To use tensor parallelism, you only need to add related information to the **parallel dict**. To be specific, `TENSOR_PARALLEL_MODE` can be '1d', '2d', '2.5d', '3d'. And the size of different parallelism should satisfy: `#GPUs = pipeline parallel size x tensor parallel size x data parallel size`.  `data parallel size` will automatically computed after you specify the number of GPUs, pipeline parallel size and tensor parallel size.

```python
from colossalai.amp import AMP_TYPE
# parallel setting
TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE)
)

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0


# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)
```

Ohter configs:
```python
# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token
```

### Start training
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_hybrid_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_hybrid_parallel.py
```

Initial parallel setting and build model:
<p align="center">
  <img src="https://github.com/yuxuan-lou/Colossal-AI-tutorials/blob/main/img/vit_hp_1.png" width="800">
</p>

During Training:
<p align="center">
  <img src="https://github.com/yuxuan-lou/Colossal-AI-tutorials/blob/main/img/vit_hp_2.png" width="800">
</p>
