# Preliminary Study on the Impact of Attention Mechanisms for Medical Image Classification

## About
Implementation of the paper [_"Preliminary Study on the Impact of Attention Mechanisms for Medical Image Classification"_](reports/Paper.pdf) by Tiago Gonçalves and Jaime S. Cardoso.

## Abstract
Despite their high performance, deep learning algorithms still work as black boxes and are not capable of explaining their predictions in a human-understandable manner, thus leading to a lack of transparency which may jeopardise the acceptance of these technologies by the healthcare community. Therefore, the topic of explainable artificial intelligence (xAI) appeared to address this issue. There are three main approaches to xAI: pre-, in- and post-model methods. In medical images, important information is generally spatially constricted. Hence, to ensure that models focus on the important parts of the images and learn relevant features, several attention mechanisms have been proposed and demonstrated increased performances. This work proposes a comparative study of the application of different attention mechanisms in deep neural networks and the evaluation of their impact on the performance of the models and the quality of the learned features.

## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:TiagoFilipeSousaGoncalves/attention-mechanisms-healthcare.git
```
Then go to the repository's main directory:
```bash
$ cd attention-mechanisms-healthcare
```

## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```

## Data
To know more about the data used in this paper, please send an e-mail to  [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt).


## Usage
### Train Models
To train the models:
```bash
$ python code/models_train.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set.
--dataset: Data set {CBIS, MIMICCXR}.
--backbone: Backbone model {densenet121, resnet50, vgg16}.
--use_attention: Use MLDAM (attention).
--nr_classes: Number of classes (using sigmoid, 1; using softmax, 2).
--gpu_id: The ID of the GPU device.
--batchsize: Batch size for dataloaders.
--epochs: Number of training epochs.
--use_daug: Use data augmentation.
```


### Test Models
To test the models:
```bash
$ python code/models_test.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set.
--dataset: Data set {CBIS, MIMICCXR}.
--nr_classes: Number of classes (using sigmoid, 1; using softmax, 2).
--gpu_id: The ID of the GPU device.
--batchsize: Batch size for dataloaders.
```


### Generate Post-hoc Explanations (Saliency Maps)
To generate post-hoc explanations (saliency maps):
```bash
$ python code/xai_generate_slmaps.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set.
--dataset: Data set {CBIS, MIMICCXR}.
--nr_classes: Number of classes (using sigmoid, 1; using softmax, 2).
--gpu_id: The ID of the GPU device.
--batchsize: Batch size for dataloaders.
```


### Generate Figures from Post-hoc Explanations (Saliency Maps)
To generate figures from post-hoc explanations (saliency maps):
```bash
$ python code/xai_generate_slmaps_figs.py {command line arguments}
```
This script accepts the following command line arguments:
```
--dataset: Data set {CBIS, MIMICCXR}.
```



## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@inproceedings{goncalves2021recpad,
	author = {Tiago Gonçalves and Jaime S. Cardoso},
	title = {{Preliminary Study on the Impact of Attention Mechanisms for Medical Image Classification}},
	booktitle = {27th Portuguese Conference in Pattern Recognition (RECPAD)},
	year = {2021},
    address = {Évora, Portugal}
}
```



## Credits and Acknowledgments
### Channel Attention Module (CAM) and Position Attention Module (PAM)
This model and associated [**code**](https://github.com/cunjian/AGPAD) are related to the paper [_"Attention-Guided Network for Iris Presentation Attack Detection"_](https://arxiv.org/abs/2010.12631) by Cunjian Chen and Arun Ross.

### Multi-Level Dual-Attention
This model and associated [**code**](https://github.com/SapnaSM/MultiLevelDAM) are related to the paper [_"Multi-Level Dual-Attention Based CNN for Macular Optical Coherence Tomography Classification"_](https://ieeexplore.ieee.org/document/8882308) by Sapna S. Mishra, Bappaditya Mandal and N. B. Puhan.
