# DSA-Model - FLICKR 8K & 30K

Model base on Show, Attend and Tell: Neural Image Caption Generation with Visual Attentiont, Soft Attention. Flickr 8k & 30K base on fuqichen implementation (https://github.com/fuqichen1998/eecs442-final-project-show-and-tell)
- CNN Layer Model: VGG16 (default)
- RNN Layer Model: LSTM (default)
- Datasets: Flickr8k (default) & Flickr30k
- Scoring: BLEU_1, BLEU_2, BLEU_3, BLEU_4 (NLTK Tools)

## Requirements
- Python 3.7.9 & Pip
- Anaconda3
- Pytorch 0.4.1 & torchvision 0.2.1
- Numpy 1.19.2
- NLTK 3.5
- h5py 2.10.0
- imageio 2.9.0
- tqdm 4.50.2
- scikit-learn 0.23.2
- scikit-image 0.17.2
- Matplotlib 3.3.2
- CUDA 9.2 (optional)
- **NOTE: IS RECOMEND TO USE A CONDA ENVIRONMENT THANKS TO THE OLD PYTORCH VERSION**


### Installation 
```
 cd dsa-model/mscoco
 conda install pytorch=0.4.1 -c pytorch
 conda install -c pytorch torchvision
 conda install -c anaconda numpy
 conda install -c conda-forge nltk
 conda install -c conda-forge h5py
 conda install -c conda-forge imageio
 conda install -c conda-forge tqdm
 conda install -c conda-forge scikit-image
 conda install -c conda-forge scikit-learn
 conda install -c conda-forge matplotlib
 ```
 
 ## Set Up
- Download & extract `flickr8k or flickr30k` (the images could be download from here: https://drive.google.com/file/d/16jNwTdwtFXoW_gsxH87TntWh6ICcFzIj/view?usp=sharing)
- Once download, set `flickr8k or flickr30k` images in the folder `img_train`
- `Flickr8k` is the default dataset, of need it it can be change to `Flickr30k` only by changing all the settings into `Flickr30k`
- Run `python dataset.py` & `python data_process.py` to configure all caption datasets

## Basic Usage
- To start traing the model run train.py all the elements are already configure

## Structure
```
├── caption_datasets/ - caption datasets
│
├── checkpoint/ - saves checkpoints of training
│
├── hypotesis/ - saves hypotesys results for internat evaluation
│
├── img_train/ - flickr images
│
├── preprocess_out/ - saves checkpoints of training
│
├── reference/ - saves references results for internat evaluation
│
├── results/ - default folder for results, saves on json
│   
├── test_caption_out/ - checks with examples final process
│
├── attentionmodel.py - helper of training class
│
├── data_process.py - configure & donwload all caption datasets
│
├── dataset.py - configure & donwload all caption datasets
│
└── evaluate.py - separate evaluation if need it checks hypotesys & reference jsons
│
└── helper.py - saves checkpoints from training
│
└── lstms.py - lstm layer of process
│
└── modelcnn.py - cnn layer of process
│
└── test_lstm.py - lstm layer of process
│
└── train.py - main class for training & evaluation
│
└── visualize.py - visualizer of progress

```

## References

* Kevin Xu, et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (https://arxiv.org/pdf/1502.03044.pdf)
* fuqichen EECS442 Final Project Winter 2019 repo (https://github.com/fuqichen1998/eecs442-final-project-show-and-tell)
