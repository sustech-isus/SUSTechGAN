# SUSTechGAN

Image Generation for Object Recognition in Adverse Conditions of Autonomous Driving

### Abstract

Autonomous driving significantly benefits from data-driven deep neural networks. However, the data in autonomous driving typically fits the long-tailed distribution, in which the critical driving data in adverse conditions is hard to collect. Although generative adversarial networks (GANs) have been applied to augment data for autonomous driving, generating driving images in adverse conditions is still challenging. In this work, we propose a novel SUSTechGAN with dual attention modules and multi-scale generators to generate driving images for improving object recognition of autonomous driving in adverse conditions. We test the SUSTechGAN and the existing well-known GANs to generate driving images in adverse conditions of rain and night and apply the generated images to retrain object recognition networks. Specifically, we add generated images into the training datasets to retrain the well-known YOLOv5 and evaluate the improvement of the retrained YOLOv5 for object recognition in adverse conditions. The experimental results show that the generated driving images by our SUSTechGAN significantly improved the performance of retrained YOLOv5 in rain and night conditions, which outperforms the well-known GANs. The open-source code, video description and datasets are available on this page to facilitate image generation development in autonomous driving under adverse conditions.

> [!CAUTION]
> **NON-STABLE VERSION WARN**
> 
> This is under review. The content in this repository may be updated.

<img src="https://github.com/sustech-isus/SUSTechGAN/assets/51916543/fb3b6404-cfba-4160-b917-5c5f134ee3d4" width="3000" height="auto" />


**IN THIS WORK**

- We design dual attention modules in SUSTechGAN to improve the local semantic feature extraction for generating driving images in adverse conditions such as rain and night. This method solves the issue that the local semantic features (e.g., vehicles) in the generated images are blurred and even approximately disappeared, and improves the object recognition of autonomous driving.
- We develop multi-scale generators in SUSTechGAN to consider various scale features (e.g., big size generator for global features and small-size generator for local features) for generating high-quality images with clear global semantic features.
- We propose a novel loss function with an extra detection loss, adversarial loss and cycle consistency loss to guide image generation for improving object recognition of autonomous driving in adverse conditions.

![image](https://github.com/sustech-isus/SUSTechGAN/assets/51916543/1ca6bc3a-66ff-4906-a334-0427d48f00b7)

<img src="https://github.com/sustech-isus/SUSTechGAN/assets/51916543/fead218d-81e9-47df-8039-dce6d3abe17f" width="3000" height="auto" />



---

### Installation

#### STEP 1: Check system prerequisites

Your system and environment meet at least the following prerequisites _(We mark the checkbox for the tested version)_

- Linux-based system
  - [x] Ubuntu 18.04 LTS / Ubuntu 20.04 LTS
- Python
  - [x] Python 3.8.10
- NVIDIA GPU with Driver 515+ and **cuDNN** base environment
  - [x] GTX 1080 Ti / Tesla V100 / Titan V

#### STEP 2: Clone this repo

```sh
cd ${YOUR_WORKSAPCE}
git clone git@github.com:sustech-isus/SUSTechGAN.git
```

#### STEP 3: Install Python requirements

```sh
cd ${REPO_ROOT}

# Install packages for base python, virtualenv, pipenv
pip install -r ./requirements.txt

# Install packages for anaconda or miniconda
conda env create -f ./requirements.yml
```

---

### Train & Test

> [!TIP]
> Visdom is included in Python package requirements, so you can use the following commands to start a vision server and view results on a web page at [http://localhost:8097]()

#### STEP 1: Get the dataset

> ![NOTE]
> Our dataset for this work is under review, and we will publish it here later!

```sh
wget -q -O - "${DATASET_URL}" | tar -xzf - -C ${REPO_ROOT}/datasets/${DATASET_NAME}
```

#### STEP 2: Begin to train

```sh
cd ${REPO_ROOT}
python train.py --dataroot ./datasets/${DATASET_NAME} --name ${DATASET_NAME} --model cyclegan
```

#### STEP 3: Begin to test

```sh
cd ${REPO_ROOT}
python test.py --dataroot ./datasets/${DATASET_NAME} --name ${DATASET_NAME} --model cyclegan
```

You can find the test results here
```
./results/${DATASET_NAME}/latest_test/index.html
```

---
### Citation

Coming soon

### About

We believe that this work is a milestone of GAN-based data generation for improving autonomous driving, and we thank the community for their support!

- [CycleGAN-and-pix2pix-in-PyTorch](https://github.com/yanqi1811/CycleGAN-and-pix2pix-in-PyTorch)
- [Pytorch-Deep Convolution GAN](https://github.com/pytorch/examples/tree/main/dcgan)
