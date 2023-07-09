## Introduction
This repository contains the official implementations of [ HVMspotter]

## Installation

First, clone the repository locally:

```shell
git clone https://github.com/whai362/pan_pp.pytorch.git
```

Then, install PyTorch 1.1.0+, torchvision 0.3.0+, and other requirements:

```shell
conda install pytorch torchvision -c pytorch
pip install -r requirement.txt
```

Finally, compile codes of post-processing:

```shell
# build pse and pa algorithms
sh ./compile.sh
```

## Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_three_map.py ${CONFIG_FILE}
```
For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/threemap_r18_ic15_container.py
```

## Testing

### Evaluate the performance

```shell
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
cd eval/
./eval_{DATASET}.sh
```
For example:
```shell
python test.py config/pan/pan_r18_ic15.py checkpoints/threemap_r18_ic15_container/checkpoint.pth.tar
cd eval/
./eval_ic15.sh
```

### Evaluate the speed

```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/threemap_r18_ic15_container/checkpoint.pth.tar --report_speed
```

## Citation

Please cite the related works in your publications if it helps your research:

###  HVMspotter

