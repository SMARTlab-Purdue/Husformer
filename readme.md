# Husformer
This repository contains the source code for our paper: "Husformer: A Multi-Modal Transformer for Multi-Modal Human State Recognition", submitted for xxxx. 
For more details, please refer to [our project website](xxxx)


## Abstract
Human state recognition is a critical topic due to its pervasive and crucial applications in human-machine systems, and multi-modal fusion that combines metrics from multiple data sources has been shown as a sound method to improve the recognition performance. In spite of the promising results of recent multi-modal-based models, they generally fail to leverage sophisticated fusion strategies that models sufficient cross-modal interactions to produce the fusion representation, and rely heavily on lengthy and inconsistent date preprocessing and feature crafting. To address these limitations, we propose an end-to-end multi-modal transformer framework for multi-modal human state recognition called Husformer. Specifically, we propose cross-
modal transformers, which inspire one modality to directly attend to latent relevance revealed in other modalities to reinforce itself, to fuse different modalities with sufficient awareness of cross-modal interactions introduced. A self-attention transformer is then utilized to further prioritize the important contextual information of human state in the fusion representation. Additionally, such two attention mechanisms enable effective and adaptive adjustments to noise and interruptions in multi-modal signals during the fusion process and at high-level feature level respectively. Extensive experiments on two human emotion (DEAP and WESAD) corpus and two cognitive workload (MOCAS and CogLoad) datasets demonstrate that our Husformer outperform state-of-the-art multi-modal baselines and the performance with a single modality in the recognition of human state by a large margin, especially when dealing with raw multi-modal signals. An ablation study is also conducted to show the benefits of each component in the Husformer.


## Overview Architecture for Husformer
<div align=center>
<img src="/figures/architecture.png" width="800" />
</div>  

## Illustration of the transformer network and multi-head self-attention

<div align=center>
<img src="/figures/CMAT.png" width="800" />
</div>  

## Main Results on Datasets with Download Address
| Name | Modalities | Acc(%) | F1(%) | #Params | Memory |  Dataset |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RAW-MOCAS | 5 | 93.71±2.26 | 93.82±2.41 | 0.74M | 210.97MB | [Raw-MOCAS](address)|
| Preprocessed-MOCAS | 5 | 96.42±2.11 | 96.51±2.03 | 0.75M | 220.53MB | [Pre-MOCAS](address) |
| RAW-DEAP(Valance) | 4 | 85.98±1.38 | 86.21±1.40 | 0.63M | 1999.56MB | [Raw-DEAP](address) |
| RAW-DEAP(Arousal) | 4 | 86.28±2.04 | 86.78±2.11 | 0.63M | 1999.56MB | [Raw-DEAP](address) |
| Preprocessed-DEAP(Valance) | 4 | 97.01±2.06 | 97.08±2.15 | 0.66M |  1773.59MB | [Pre-DEAP](address) |
| Preprocessed-DEAP(Arousal) | 4 | 97.67±1.45 | 97.69±1.53 | 0.66M |  1773.59MB | [Pre-DEAP](address) |
| WESAD | 4 | 85.02±1.91 | 85.85±2.14 | 0.71M | 3084.35MB | [WESAD](address) |
| Cogload | 4 | 80.40±2.34 | 81.27±2.63 | 0.72M | 94.28MB | [Cogload.pkl](data/)|


## Usage
### Requirements
1. Install the required Python package and version

- Python 3.8
- [Pytorch (1.8.2+cu111) and torchvision or above](https://pytorch.org/)
- CUDA  11.1 or above
- scikit-learn  1.0.2
- numpy 1.19.5

### Datasets
Data files (containing processed MOCAS, DEAP,Cogload and WESAD datasets) can be downloaded from table above.

#### Make data file as '.pkl'
Husformer read data from 'husformer.pkl' in [data/](data/).
You should change data file format from '.xxx' to '.pkl', before you start run training or testing command.
We provide Python files to change data file format in [make_data](make_data/), and name them as dataset's name,such as: [Pre-MOCAS.py](make_data/Pre-MOCAS.py) and [Raw-MOCAS.py](make_data/Raw-MOCAS.py).
You should add the datasets file path to 'dataset_name_list.txt' after download datasets, and change the concant as 'dataset_name_list.txt' in 'dataset_name.py'.
For each dataset, we randomly shuffled all data and conducted the K-folder Cross Validation (K = 10). So you will get 10 '.pkl' files every time after you run the code below. 

### Run the code
0. We provide [cogload.pkl](data/) in [data](data/). You should make data as follow, if you want to use other datasets.
```
python make_data/dataset_name.py
```
then put the prepared '.pkl' data in 'data/', and rename it as 'husformer.pkl'.

1. Training command as follow. 
```
python main.py
```

2. Testing command as follow.
```
python main.py --eval
```


(The code was tested in Ubuntu 18.04 with Python 3.8.)
### Change model
We provide 3 models in code to correspond to the 3 data number of modalities as 3,4,5. You should add or remove some modalities in code, if you want to use more or less modalities by Husformer.
#### How change the model
We name these files with their modalities. We put the general python files in [src](src), as the same time, the different modalities main python files are [main-3.py](main-3.py), [main-4.py](main-4.py), [main-5.py](main-5.py) and different modalities python files in folder [src/3](src/3),[src/4](src/4) and [src/5](src/5), you should rename the main python files as 'main.py' and move the python files from 'src/x' to [src](src/).


## Citation
If you find the code or the paper useful for your research, please cite our paper:
```

```

## Acknowledgement

Contributors:  
[Ruiqi Wang](https://github.com/R7-Robot);[Dezhong Zhao](https://github.com/zdz0086).

Part of the code is based on the following repositories:  

[Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer).




