## [RMFDNet: Redundant and Missing Feature Decoupling Network for Salient Object Detection]
by Qianwei Zhou, Jintao Wang, Jiaqi Li, Chen Zhou, Haigen Hu, Keli Hu

# Introduction
Recently, many salient object detection methods have utilized edge contours to constrain the solution space. This approach aims to reduce the omission of salient features and minimize the inclusion of non-salient features. To further leverage the potential of edge-related information, this paper proposes a Redundant and Missing Feature Decoupling Network (RMFDNet). RMFDNet primarily consists of a segment decoder, a complement decoder, a removal decoder, and a recurrent repair encoder. The complement and removal decoders are designed to directly predict the missing and redundant features within the segmentation features. These predicted features are then processed by the recurrent repair encoder to refine the segmentation features. Experimental results on multiple Red-Green-Blue (RGB) and Red-Green-Blue-Depth (RGB-D) benchmark datasets, as well as polyp segmentation datasets, demonstrate that RMFDNet significantly outperforms previous state-of-the-art methods across various evaluation metrics. The efficiency, robustness, and generalization capability of RMFDNet are thoroughly analyzed through a carefully designed ablation study. The code will be made available upon paper acceptance.

## Prerequisites
- install miniconda
- $ conda env create -f environment.yaml

## Clone repository
```shell
git clone https://github.com/QianWeiZhou/RMFDNet.git
```

## Download dataset
Download the following datasets and unzip them into `data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)
- [NJU2K](https://drive.google.com/open?id=1R1O2dWr6HqpTOiDn6hZxUWTesOSJteQo)
- [NLPR](https://sites.google.com/site/rgbdsaliency/dataset)
- [SIP](https://drive.google.com/open?id=1R91EEHzI1JwfqvQJLmyciAIWU-N8VR4A)
- [STERE](https://drive.google.com/file/d/1JYfSHsKXC3GLaxcZcZSHkluMFGX0bJza/view?usp=sharing)
- [DTU-D](https://drive.google.com/open?id=1DzkswvLo-3eYPtPoitWvFPJ8qd4EHPGv)

## Training & Evaluation
- If you want to train the model by yourself, please download the [pretrained model](https://download.pytorch.org/models/resnet50-19c8e357.pth) into `res` folder
- If you want to train the model with RGB dataset
```shell
    cd RMFDNet_RGB/
    python3 train.py
```
- If you want to train the model with RGBD dataset
```shell
    cd RMFDNet_RGBD/
    python3 train.py
```

## Testing 
- Predict the saliency maps
```shell
    python3 test.py
```

## Saliency maps & Trained model
- saliency maps & trained model: https://pan.baidu.com/s/12Q3mhYPw_zqjUIE5rFg2SQ?pwd=bgky code：bgky 



## Citation
- If you find this work is helpful, please cite our paper
```

```
