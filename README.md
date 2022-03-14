# RecNS
Source code and dataset for TKDE'22 paper "Region or Global? A Principle for Negative Sampling in Graph-based Recommendation"


Region or Global? A Principle for Negative Sampling in Graph-based Recommendation

Zhen Yang, Ming Ding, Xu Zou, Jie Tang, Fellow,IEEE, Bin Xu, Chang Zhou, and Hongxia Yang

In TKDE 2022 


## Introduction
RecNS is a general negative sampling method designed with two sampling strategies: positive-assisted sampling and exposure-augmented sampling, which utilize the proposed Three-Region Principle to guide negative sampling. The Three-Region Principle suggests that we should negatively sample more items at an intermediate region and less adjacent and distant items. 

## Preparation
* Python 3.7
* Tensorflow 1.14.0


## Training
### Training on the existing datasets
#### For PinSage:
You can use ```$ ./experiments/***.sh``` to train RecNS model. For example, if you want to train on the Zhihu dataset, you can run ```$ ./experiments/recns_zhihu.sh ``` to train RecNS model.

#### For LightGCN:
You can use ```$ ./***.sh``` to train RecNS model. For example, if you want to train on the Zhihu dataset, you can run ```$ ./train.sh``` to train RecNS model.


## Cite
Please cite our paper if you find this code useful for your research:
```
@article{yang2022region,
  title={Region or Global A Principle for Negative Sampling in Graph-based Recommendation},
  author={Yang, Zhen and Ding, Ming and Zou, Xu and Tang, Jie and Xu, Bin and Zhou, Chang and Yang, Hongxia},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```
