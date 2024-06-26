# JMAN
This is official Pytorch implementation of "Efficient Pansharpening by Joint-modality Recursive
Training"

## Framework
<div>
    <img src="https://github.com/songwenhao123/JMAN/blob/main/JMAN/figure/pans.jpg" alt="Framework" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The overall framework of the proposed JMAN.</em>
</p>

## Introduction
Multispectral images captured by remote sensing
systems usually have shallow spatial resolution. Pansharpening
offers a promising solution by enhancing the resolution of these
Low-resolution Multi-Spectral (LMS) images to a High-resolution
Multi-Spectral (HMS) without the need for costly hardware
upgrades. Unfortunately, existing pansharpening models achieve
impressive results with numerous learnable parameters, which
makes them impractical for integration into remote sensing
systems. Moreover, the existing methods suffer from utilizing
only the CNN model while ignoring the global information of
the image. In this work, a parameter-efficient pansharpening
model, named Joint-Modality Association Network (JMAN), is
built by leveraging complementary information from multiple
modalities and recursive training. It has successfully improved
the resolution of remote sensing images, enabling them to have
broad applications in environmental monitoring and assessment.
Specifically, we efficiently leverage the complementary informa-
tion from different networks, including the transformer and
convolutional neural network (CNN), and use hierarchical asso-
ciation mechanisms to create a more distinctive and informative
representation by associating intra-modality and cross-modality.
Furthermore, the parameter-sharing mechanism of recursive
training can effectively reduce the number of parameters in
the model. Benefiting from its lightweight design and effective
information fusion strategy, the proposed method can generate
faithful pansharpened multispectral images that excel in both
spectral and spatial resolution. Experimental results show the
superiority of the proposed method over extensive benchmarks.

## Recommended Environment
* Python3.7+, Pytorch>=1.6.0
* NVIDIA GPU + CUDA
* Run `python setup.py develop`

## Quick Start
**Step0. Set your Python environment.**

>git clone https://github.com/songwenhao123/JMAN(Pytorch)

Then, 

`python setup.py develop`

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
`python run_pansharpening.py`

**Step3. To test**
`python run_test_pansharpening.py`

