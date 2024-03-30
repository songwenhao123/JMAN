# JMAN
This is official Pytorch implementation of "Efficient Pansharpening by Joint-modality Recursive
Training"

## Framework
<div>
    <img src="https://github.com/songwenhao123/JMAN/blob/main/JMAN/figure/overview.pdf" alt="Framework" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The overall framework of the proposed JMAN.</em>
</p>

## Introduction


## Recommended Environment
* Python3.7+, Pytorch>=1.6.0
* NVIDIA GPU + CUDA
* Run `python setup.py develop`
    
## To Test
1. Downloading the pre-trained checkpoint from [best_model.pth](https://pan.baidu.com/s/1N_dZvfiKwuwQf2DZPstJ0A?pwd=PSFu) and putting it in **./results/PSFusion/checkpoints**.
2. Downloading the MSRS dataset from [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.
3. `python test_Fusion.py --dataroot=./datasets --dataset_name=MSRS --resume=./results/PSFusion/checkpoints/best_model.pth`

If you need to test other datasets, please put the dataset according to the dataloader and specify **--dataroot** and **--dataset-name**

## Quick Start
**Step0. Set your Python environment.**

>git clone https://github.com/songwenhao123/JMAN(Pytorch)

Then, 

> python setup.py develop

**Step1. Put datasets and set path**
* Put datasets (WorldView-3, QuickBird, GaoFen2, WorldView2) into the `UDL/Data/pansharpening`, see following path structure. 

```
|-$ROOT/Data
├── pansharpening
│   ├── training_data
│   │   ├── train_wv3.h5
│   │   ├── ...
│   ├── validation_data
│   │   │   ├── valid_wv3.h5
│   │   │   ├── ...
│   ├── test_data
│   │   ├── WV3
│   │   │   ├── NY1_WV3_RR.mat
│   │   │   ├── ...
│   │   │   ├── ...
```
**Step2. To train**
> python run_pansharpening.py

**Step3. To test**

> python run_test_pansharpening.py

