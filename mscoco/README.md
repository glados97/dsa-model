# DSA-Model - MSCOCO

Model base on Show, Attend and Tell: Neural Image Caption Generation with Visual Attentiont, Soft Attention.
Mscoco model base on coldmanck implementation (https://github.com/coldmanck/show-attend-and-tell)
- CNN Layer Model: VGG16 (default)
- RNN Layer Model: LSTM (default)
- Dataset: MS-COCO (default)
- Scoring: BLEU_1, BLEU_2, BLEU_3, BLEU_4, METEOR, ROUGE_L, CIDEr (Microsoft COCO Caption Evaluation by Tsung-Yi Li implementation (https://github.com/tylin/coco-caption))

## Requirements
- Python 3.6 & Pip
- Tensorflow 1.14 (for more information check: https://www.tensorflow.org/install/pip?hl=es-419&lang=python3#windows)
- Numpy 1.19.2
- NLTK 3.5
- tqmd 4.50.2
- Pandas 1.1.3
- Pycocotools 2.0.2
- Pickle5 0.0.11
- Matplotlib 3.3.2
- Opencv 4.2.0.34
- CUDA 9.2 (optional)
- Java

### Installation 
```
 cd dsa-model/mscoco
 pip install --user --upgrade tensorflow  
 pip install numpy
 pip install nltk
 pip install tqdm
 pip install pycocotools
 pip install pickle5
 pip install matplotlib
 pip install opencv-python
 sudo apt install default-jdk
```

## Set Up
- Download & extract mscoco `train2014`, `val2014` & `annotations` (the images could be download from here: https://drive.google.com/file/d/16jNwTdwtFXoW_gsxH87TntWh6ICcFzIj/view?usp=sharing)
- Once download, set `train2014` images in the folder `train/images`, and put the file `captions_train2014.json` in the folder `train`; And the `val2014` images in the folder `val/images`, and put the file `captions_val2014.json` in the folder `val`.
- Download the pretrained VGG16 net [here](https://app.box.com/s/idt5khauxsamcg3y69jz13w6sc6122ph)
- Run python `config.py`

## Basic Usage
- To start traing the model run main.py with the following elements:
```
python main.py --phase=train \
    --load_cnn \
    --cnn_model_file='./vgg16_no_fc.npy'\
    [--train_cnn]  
 ```
 -  Once finish, to start evaluating the model run main.py with the following elements:
```
python main.py --phase=eval \
    --model_file='./models/xxxxxx.npy' \
    --beam_size=3
```
- **NOTE: check the model to evaluate depending on the epoch **

## Structure
```
├── examples/ - coco images examples
│
├── models/ - train models saves in here
│
├── results/ - default folder for results, saves on json
│
├── summary/ - tensorBoard visualization
│
├── test/ - models, losses, and metrics
│   ├── images/ - clean images 
│   └── results/ - examples of images with description
│
├── train/ - models, losses, and metrics
│   ├── image/ - train2014 images
│   └── captions_train2014.json - train captions
│
├── utils/ - pycoco evaluation
│   
├── val/ - models, losses, and metrics
│   ├── image/ - val2014 images
│   └── captions_val2014.json - val captions
│
├── base_model.py - base model
│
├── config.py - preconfiguration for training
│
├── dataset.py - checks the images and send it to the model
│
└── main.py - main class for training & evaluation
│
└── model.py - trains the model

```

## References

* coldmanck Python 3 Version of Show, Attend and Tell using Tensorflow repo (https://github.com/coldmanck/show-attend-and-tell)
* Lin, Tsung-Yi Microsoft COCO Caption Evaluation repo (https://github.com/tylin/coco-caption)
