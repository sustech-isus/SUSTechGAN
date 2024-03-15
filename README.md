# SUSTechGAN

Image Generation for Object Recognition under Adverse Conditions of Autonomous Driving

### Introduction

SUSTechGAN is a GAN-based image generation toolkit with dual attention modules and multi-scale generators to generate driving images for improving object recognition of autonomous driving in adverse conditions.

> [!CAUTION]
> **NON-STABLE VERSION WARN**
> 
> The paper of this work is under reviewing. The feature contained in this repository is subject to change at any time.

![278d4422-69e7-4013-83e4-16eed546a5cf](https://github.com/sustech-isus/SUSTechGAN/assets/51916543/62f032a3-58a5-4170-9bbd-fa0f7fc894b2)


**IN THIS WORK**

- We design dual attention modules in SUSTechGAN to improve the local semantic feature extraction for generating driving images in adverse conditions such as rain and night. This method solves the issue that the local semantic features (e.g., vehicles) in the generated images are blurred and even approximately disappeared, and improves the object recognition of autonomous driving.
- We develop multi-scale generators in SUSTechGAN to consider various scale features (e.g., big size generator for global feature and small size generator for local feature) for generating high-quality images with clear global semantic features.
- We propose a novel loss function with an extra detection loss, adversarial loss and cycle consistency loss to guide image generation for improving object recognition of autonomous driving in adverse conditions.

![image](https://github.com/sustech-isus/SUSTechGAN/assets/51916543/1ca6bc3a-66ff-4906-a334-0427d48f00b7)


---

### Installation

#### STEP 1: Check system prerequesties

Your system and environment meets at least the following prerequesties _(We mark checkbox for tested version)_

- Linux based system
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

#### STEP 3: Install python requirments

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
> Visdom is included in python package requirements, so you can use the follow commands to start a  visdom server and view results in a web page at [http://localhost:8097]()

#### STEP 1: Get dataset

> ![NOTE]
> Our dataset for this work is under review and we will publish it here later!

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

You can find test result here
```
./results/${DATASET_NAME}/latest_test/index.html
```

---

### About

This work is just one step closer to the application of GAN to autonomous driving, and we thank the community for their support on this work!

- [CycleGAN-and-pix2pix-in-PyTorch](https://github.com/yanqi1811/CycleGAN-and-pix2pix-in-PyTorch)
- [Pytorch-Deep Convolution GAN](https://github.com/pytorch/examples/tree/main/dcgan)
