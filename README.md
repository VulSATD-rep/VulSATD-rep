# VulSATD replication package

This repository contains the replication package for the paper "VulSATD: leveranging  machine learning to detect vulnerable self-admitted technical debt". 

## How to replicate

The first step is to clone the repository locally with the following commands:

```
git clone https://github.com/VulSATD/VulSATD-rep
cd VulSATD-rep
```

We recommend the creation of a virtual environment, for example, using venv (the exact commands might need to be adapted according to your system):

```
python3 -m venv env
source env/bin/activate
```

Then, install the dependencies which are listed in the requirements.txt file. This can be done using `pip`:

```
pip install -r requirements.txt
```

## Datasets

The datasets used in this study are:

|**Dataset**|**Projects**|
|-------|--------|
|WeakSATD    |Chromium, Mozilla Firefox and Linux Kernel |
|Devign |QEMU and FFmpeg | 
|BigVul |Chromium, Linux Kernel, Android, FFmpeg, php, ImageMagick, Radare2, Kerberos 5, and Tcpdump |  

Each dataset file contains, at least, the following columns:

|**Column**|**Description**|
|-------|--------|
|Code   |The function code extended with previous comments if existent. |
|Coments|The comments extracted from the extended function code. |
|OnlyCode | The extended function code with the comments removed.
|SATD   |Label describing if one or more comments in the code are SATD. |
|Vulnerable|Label describing if the code is vulnerable. |

For each dataset, we provide two folders containing the split into train, validation, and test used in the study:

- one folder containing the complete version
- one folder containing only functions with comments

To download the folder containing all the datasets needed for replicating the study, use the following command:

```
gdown --folder https://drive.google.com/drive/folders/1gxEPuqnMGz3KbnN3Tmf2k-Eb60Rey4rN
```

## RQ1: VulSATD detection


To train the model, use the following command:

```
python main.py \
        --model=vulsatd \
        --mode=train \
        --dataset="datasets/weaksatd-commented-only" \
        --store-weights=True \
        --output-dir="./stored_models/weaksatd-commented-only"
```

To test the saved model, use the following command:

```
python main.py \
       --model=vulsatd \
       --mode=test \
       --dataset="datasets/weaksatd-commented-only" \
       --model-file="./stored_models/weaksatd-commented-only/weights_vulsatd_lr_2e-05_ne_10_bs_16_dp_0.1_l2_0.0.tf"
```

The previous commands should be repeated replacing `weaksatd` for the folders of the other datasets:

|Folder|Dataset|
|------|-------|
|weaksatd-commented-only|WeakSATD|
|devign-commented-only|Devign|
|bigvul-10-commented-only|BigVul|



## RQ2: detection of SATD and vulnerable code (multitask)

To train the model for multitask, use the following command:

```
python main.py \
        --model=multitask \
        --mode=train \
        --dataset="datasets/weaksatd-commented-only" \
        --store-weights=True \
        --output-dir="./stored_models/weaksatd-commented-only"
```

To test the saved model, use the following command:

```
python main.py \
       --model=multitask \
       --mode=test \
       --dataset="datasets/weaksatd-commented-only" \
       --model-file="./stored_models/weaksatd-commented-only/weights_multitask_lr_2e-05_ne_10_bs_16_dp_0.1_l2_0.0.tf"
```

Then, to run the model for the single tasks (SATD only and vulnerable only), the commands are:

```
python main.py \
        --model=satdonly \
        --mode=train \
        --dataset="datasets/weaksatd-commented-only" \
        --store-weights=True \
        --output-dir="./stored_models/weaksatd-commented-only"
```

To test the saved model, use the following command:

```
python main.py \
       --model=satdonly \
       --mode=test \
       --dataset="datasets/weaksatd-commented-only" \
       --model-file="./stored_models/weaksatd-commented-only/weights_satdonly_lr_2e-05_ne_10_bs_16_dp_0.1_l2_0.0.tf"
```



```
python main.py \
        --model=vulonly \
        --mode=train \
        --dataset="datasets/weaksatd-commented-only" \
        --store-weights=True \
        --output-dir="./stored_models/weaksatd-commented-only"
```

To test the saved model, use the following command:

```
python main.py \
       --model=vulonly \
       --mode=test \
       --dataset="datasets/weaksatd-commented-only" \
       --model-file="./stored_models/weaksatd-commented-only/weights_vulonly_lr_2e-05_ne_10_bs_16_dp_0.1_l2_0.0.tf"
```

