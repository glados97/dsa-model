import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CustomDataset(Dataset):
    def __init__(self, path, dataset, split, transform=None):
        """
        :param path: path to preprocessed files
        :param dataset: Can be one of 'coco', 'flickr8k', 'flickr30k'
        :param split: 'TRAIN', 'VAL', or 'TEST'
        """
        self.transform = transform
        self.split = split
        h5_file = h5py.File(os.path.join(path, self.split + '_IMAGES_' + dataset + '.hdf5'), 'r')
        self.caps_per_img = h5_file.attrs['caps_per_img']
        self.images_data = h5_file['images_data']

        with open(os.path.join(path, self.split + '_CAPTIONS_' + dataset + '.json'), 'r') as j:
            self.captions_indexed = json.load(j)

        with open(os.path.join(path, self.split + '_CAPTION_LEN_' + dataset + '.json'), 'r') as j:
            self.cap_len = json.load(j)

        self.dataset_size = len(self.captions_indexed)


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, i):
        img_idx = i // self.caps_per_img
        # NOTE: Each image has caps_per_img number of captions
        img = torch.FloatTensor(self.images_data[img_idx] / 255.0)
        if self.transform:
            img_normalized = self.transform(img)
        caption = torch.LongTensor(self.captions_indexed[i])
        cap_len = torch.LongTensor([self.cap_len[i]])

        # For training, only one caption needs to be returned and for val/test
        # we need to also return all captions for this image for calculating scores
        if self.split is 'TRAIN':
            return img, caption, cap_len
        elif self.split is 'VAL':
            all_captions = torch.LongTensor(self.captions_indexed[(img_idx * self.caps_per_img):
                                                                  (img_idx * self.caps_per_img + self.caps_per_img)])
            return img, caption, cap_len, all_captions
        else:
            all_captions = torch.LongTensor(self.captions_indexed[(img_idx * self.caps_per_img):
                                                                  (img_idx * self.caps_per_img + self.caps_per_img)])
            return img_normalized, img, caption, cap_len, all_captions


# if __name__== "__main__":
#     train_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TRAIN'),
#                                                batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
#     val_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'VAL'),
#                                                batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
#     for i, data in enumerate(train_loader):
#         img, caption, cap_len = data
#         # print(caption[0], cap_len[0])
#         print(cap_len.shape)
#         break
