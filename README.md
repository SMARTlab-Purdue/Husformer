# Husformer
This repository contains the source code for our paper: "Husformer: A Multi-Modal Transformer for Multi-Modal Human State Recognition".
For more details, please refer to [our paper](https://arxiv.org/abs/2209.15182).


## Abstract
Human state recognition is a critical topic due to its pervasive and crucial applications in human-machine systems, and multi-modal fusion that combines metrics from multiple data sources has been shown as a sound method to improve the recognition performance. In spite of the promising results of recent multi-modal-based models, they generally fail to leverage sophisticated fusion strategies that models sufficient cross-modal interactions to produce the fusion representation, and rely heavily on lengthy and inconsistent date preprocessing and feature crafting. To address these limitations, we propose an end-to-end multi-modal transformer framework for multi-modal human state recognition called Husformer. Specifically, we propose cross-modal transformers, which inspire one modality to directly attend to latent relevance revealed in other modalities to reinforce itself, to fuse different modalities with sufficient awareness of cross-modal interactions introduced. A self-attention transformer is then utilized to further prioritize the important contextual information of human state in the fusion representation. Additionally, such two attention mechanisms enable effective and adaptive adjustments to noise and interruptions in multi-modal signals during the fusion process and at high-level feature level respectively. Extensive experiments on two human emotion (DEAP and WESAD) corpus and two cognitive workload (MOCAS and CogLoad) datasets demonstrate that our Husformer outperform state-of-the-art multi-modal baselines and the performance with a single modality in the recognition of human state by a large margin, especially when dealing with raw multi-modal signals. An ablation study is also conducted to show the benefits of each component in the Husformer.

## Overview

### Overall Architecture of our Husformer
<div align=center>
<img src="/figures/architecture.png" width="800" />
</div>  

### Architectural description of (a) cross-modal attention and (b) cross-modal transformer network

<div align=center>
<img src="/figures/CMAT.png" width="800" />
</div>  





## Usage

### Prerequisites

- Python 3.8
- [Pytorch (1.8.2+cu111) and torchvision](https://pytorch.org/)
- CUDA 11.1 or above
- Scikit-learn 1.0.2
- Numpy 1.19.5

(The code was tested in Ubuntu 18.04 with Python 3.8.)

### Datasets


[MOCAS](https://polytechnic.purdue.edu/ahmrs/mocas-dataset) | [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) |[WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) | [Cogload](https://gitlab.fri.uni-lj.si/lrk/mobile-cogload-dataset)

Downloading addresses of datasets including DEAP, WESAD, MOCAS and CogLoad can be found above.

#### Convert the data file format to '.pkl'

Husformer reads and loads data from 'Husformer.pkl' in [data/](data/) for training and testing.

Before starting to run the training or testing commands, you should convert the data file format from '.xxx', e.g., '.csv', to '.pkl', and rename the data file as 'Husformer.pkl'.

We provide Python code demos used for data format converting in [make_data](make_data/), and name them as 'dataset's name.py', such as: [Pre-MOCAS.py](make_data/Pre-MOCAS.py) and [Raw-MOCAS.py](make_data/Raw-MOCAS.py). You should create a 'dataset_name_list.txt' with the downloaded dataset file path contained for the make_data codes to locate the data file.

For each dataset, we randomly shuffled all data and conducted the K-folder Cross Validation (K = 10). Thus you will get 10 '.pkl' files every time after running the make_data code. 

### Change the model file according to the number of modalities

We provide 3 model files which are corresponding to task scenarios involving 3, 4, and 5 modalities. You can follow the provided demos to make new model files if you want to use more or fewer modalities with the Husformer.

#### How to change the model files

1. Move the target model files contained in folders, e.g., [src/3](src/3), [src/4](src/4) and [src/5](src/5), from 'src/x' to [src](src/).

2. Rename the target 'main-x.py' in [src](src), e.g., [main-3.py](main-3.py), [main-4.py](main-4.py), [main-5.py](main-5.py), as 'main.py'.

### Recommend Hyper-parameters on each dataset

|Parameter Name|Raw DEAP|Preprocessed DEAP|WESAD|Raw MOCAS|Preprocessed MOCAS|Cogload|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Batch Size|1024|1024|512|64|128|1024|
|Initial Learning Rate|2e-3|2e-3|1e-3|1e-3|1e-3|1e-3|
|Optimizer|SGD|SGD|Adam|Adam|Adam|Adam|
|Transformer Hidden Unit Size|40|40|40|40|40|40|
|Crossmodal Attention Heads|3|3|3|5|5|3|
|Crossmodal Attention Block Dropout|0.1|0.1|0.05|0.05|0.05|0.1|
|Output Dropout|0.1|0.1|0.1|0.1|0.1|0.1|
|Focal Loss $\alpha_c$|[0.1,0.1,0.8]|[0.15,0.05,0.8]|[0.4,0.3,0.3]|[0.2,0.1,0.7]|[0.15,0.15,0.7]|[0.1,0.1,0.8]|
|Focal Loss $\gamma$|3|3|2|3|3|2|
|Epochs|20|20|60|40|40|80|

### Run the code

0. We provide converted [cogload.pkl](data/) in [data](data/). You should make data as following if using other datasets.
```
python make_data/dataset_name.py
```
Then put the made '.pkl' data file in 'data/', and rename it as 'Husformer.pkl'.

1. Training command as follow. 
```
python main.py
```

2. Testing command as follow.
```
python main.py --eval
```




## Citation

If you find the code or the paper useful for your research, please cite our paper:
```
@article{wang2022husformer,
  title={Husformer: A Multi-Modal Transformer for Multi-Modal Human State Recognition},
  author={Wang, Ruiqi and Jo, Wonse and Zhao, Dezhong and Wang, Weizheng and Yang, Baijian and Chen, Guohua and Min, Byung-Cheol},
  journal={arXiv preprint arXiv:2209.15182},
  year={2022}
}
```

## Acknowledgement

Contributors:  
[Ruiqi Wang](https://github.com/R7-Robot); [Dezhong Zhao](https://github.com/zdz0086); [Wonse Jo](http://www.smart-laboratory.org/group/Wonse_Jo.html); [Byung-Cheol Min](http://www.smart-laboratory.org/group/bcm.html).

Part of the code is based on the following repositories:  
[Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer).


## Notes
Please note that the code available on GitHub has not been updated with our latest experimental configurations; this update will occur once our paper is officially published.

