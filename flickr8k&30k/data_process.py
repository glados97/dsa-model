import numpy as np
import os
from tqdm import tqdm
import json
from scipy.misc import imresize
from imageio import imread
from collections import Counter
import random
import h5py


# This data loading file is derived from public kaggle kernel: https://www.kaggle.com/rohitag13/create-data-imagecaptioning
def process_images_captions(dataset='coco', cap_json_path='./caption_datasets/dataset_coco.json', img_path='./img_train',
                            out_path='./preprocess_out', min_word_freq=5, max_cap_len=80, caps_per_img=5,
                            img_out_dimension = (256, 256)):
    """
    Process the image datasets and also the caption json file associated with that image datasets.
    The default dataset is coco, but also supports flickr 8k and flickr 30k
    Output Files:
    1. HDF5 file containing images for each train, val, test in an I, 3, 256, 256 tensor.
       Pixels are unsigned 8 bit integer in range [0, 255]
    2. JSON file with encoded captions
    3. JSON file with caption lengths
    4. JSON file with word embedding map

    :param dataset: Can be one of 'coco', 'flickr8k', 'flickr30k'
    :param cap_json_path: JSON file that preprocesses image caption labels, see readme file for details
    This JSON file is downloaded from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    :param img_path: path that contain all training images. If using coco data set, make sure immg_path contains two
    sub folders "train2014/" and "val2014/" which will be the COCO data downloaded from COCO website
    :param out_path: ouput files will be written to this folder
    :param min_word_freq: The minimum word frequency to be considered in wordmap, if smaller than min_word_freq
    <unk> token will be used
    :param max_cap_len: only consider captions shorter than max caption length
    :param caps_per_img: Captions per image that will be compiled to output file
    :return: No return. All preprocessed files will be written to out_path
    """
    train_paths = []
    train_caps = []
    val_paths = []
    val_caps = []
    test_paths = []
    test_caps = []
    # Counter usage: https://pymotw.com/2/collections/counter.html
    word_freq = Counter()
    data = None

    with open(cap_json_path, 'r') as j:
        data = json.load(j)

    for img in data['images']:
        # caps example : [['a', 'wolverine'], ['a', 'b'], ['c']]
        cap = []
        path = None
        for sentence in img['sentences']:
            # Append sentence to caption
            if max_cap_len >= len(sentence['tokens']):
                cap.append(sentence['tokens'])
            # Update word frequency count
            word_freq.update(sentence['tokens'])

        # If this image has no captions, skip the rest of the storing process
        if len(cap) == 0:
            continue

        # Distribute captions and paths to train, validation, test
        if dataset == 'coco':
            path = os.path.join(img_path, img['filepath'], img['filename'])
        else:
            path = os.path.join(img_path, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_paths.append(path)
            train_caps.append(cap)
        elif img['split'] in {'val'}:
            val_paths.append(path)
            val_caps.append(cap)
        elif img['split'] in {'test'}:
            test_paths.append(path)
            test_caps.append(cap)

    # Create word map
    # qualified_words = []
    word_map = {'<pad>': 0}
    idx = 1
    for key in word_freq.keys():
        if word_freq[key] > min_word_freq:
            word_map[key] = idx
            idx += 1
    word_map['<unk>'] = idx
    idx += 1
    word_map['<start>'] = idx
    idx += 1
    word_map['<end>'] = idx

    with open(os.path.join(out_path, 'DICTIONARY_WORDS_' + dataset + '.json'), 'w') as f:
        json.dump(word_map, f)

    random.seed(442)
    process_img(train_paths, train_caps, 'TRAIN', out_path, dataset, caps_per_img, word_map, max_cap_len, img_out_dimension)
    process_img(val_paths, val_caps, 'VAL', out_path, dataset, caps_per_img, word_map, max_cap_len, img_out_dimension)
    process_img(test_paths, test_caps, 'TEST', out_path, dataset, caps_per_img, word_map, max_cap_len, img_out_dimension)


def process_img(img_paths, img_caps, split, out_path, dataset, caps_per_img, word_map, max_cap_len, img_out_dimension):
    with h5py.File(os.path.join(out_path, split + '_IMAGES_' + dataset + '.hdf5')) as h5_file:
        print("\nNow Writing to h5py file " + split + '_IMAGES_' + dataset + ".hdf5 with " + str(len(img_paths)) + " images:")

        h5_file.attrs['caps_per_img'] = caps_per_img
        h5_file.attrs['max_cap_len'] = max_cap_len

        h5_img_dataset = h5_file.create_dataset('images_data', (len(img_paths), 3, img_out_dimension[0], img_out_dimension[1]), dtype='uint8')

        captions_indexed = []
        cap_len = []

        for idx, path in enumerate(tqdm(img_paths)):
            if len(img_caps[idx]) < caps_per_img:
                captions = img_caps[idx] + random.choice(img_caps[idx], caps_per_img - len(img_caps[idx]))
            else:
                captions = random.sample(img_caps[idx], k=caps_per_img)

            # Read images
            img = imread(img_paths[idx])
            img = imresize(img, (img_out_dimension[0], img_out_dimension[1], 3)).transpose(2, 0, 1)
            # print(img.shape)
            # Save image to HDF5 file
            # assert img.shape == (3, 256, 256)
            h5_img_dataset[idx] = img

            for cap in captions:
                # <start> + tokens + <end> + <pad> * padding_length
                indexed_cap = []
                indexed_cap.append(word_map['<start>'])
                indexed_cap = indexed_cap + [word_map.get(word, word_map['<unk>']) for word in cap]
                indexed_cap.append(word_map['<end>'])
                indexed_cap = indexed_cap + [word_map['<pad>']] * (max_cap_len - len(cap))
                captions_indexed.append(indexed_cap)
                # add 2 due to start and end
                cap_len.append(len(cap) + 2)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(out_path, split + '_CAPTIONS_' + dataset + '.json'), 'w') as j:
            json.dump(captions_indexed, j)

        with open(os.path.join(out_path, split + '_CAPTION_LEN_' + dataset + '.json'), 'w') as j:
            json.dump(cap_len, j)


if __name__== "__main__":
    process_images_captions('flickr8k', './caption_datasets/dataset_flickr8k.json')
