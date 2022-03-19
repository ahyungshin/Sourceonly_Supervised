import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2

<<<<<<< HEAD
class cityscapesDataSet(data.Dataset):
    def __init__(self, root = "/root/agilab22/3CUDA/Cityscapes/data", list_path="./dataset/cityscapes_list/train.txt", max_iters=None, crop_size=(1280,720), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
=======

class cityscapesDataSet(data.Dataset):
    def __init__(self, root = "", list_path="", max_iters=None, crop_size=(1280,720), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
>>>>>>> target supervised
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
<<<<<<< HEAD
            self.files.append({
                "img": img_file,
                "name": name
            })

=======
            label_file = osp.join(self.root, "gtFine/%s/%s_gtFine_labelIds.png" % (self.set, name[:-16])) # leftImg8bit.png -> gtFine_labelIds.png
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        # class 19
>>>>>>> target supervised
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

<<<<<<< HEAD
=======
        # class 13
        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6,
        #                       24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}



>>>>>>> target supervised
    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
<<<<<<< HEAD
        label = datafiles["img"][:-15]+"gtFine_labelIds.png"
        label = label.split('/')
        label[6] = 'gtFine'
        label = (label[0] + '/' + label[1] +'/' + label[2] + '/' + label[3] + '/' + label[4] + '/' + label[5] + '/' + label[6] + '/'  
                + label[7] + '/' + label[8] + '/' + label[9])
        image = Image.open(datafiles["img"]).convert('RGB')
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        name = datafiles["name"]

        # resiz
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = cv2.resize(label, (1024,512), Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label = self.id2trainId(label)
=======

        label = datafiles["label"]
        if set != 'train':
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (1024,512), Image.NEAREST)
            label = np.asarray(label, np.float32)
            label = self.id2trainId(label)

        name = datafiles["name"]

        image = Image.open(datafiles["img"]).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)
>>>>>>> target supervised
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
<<<<<<< HEAD
        #label = label.transpose((2, 0, 1))
=======
        
>>>>>>> target supervised
        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesDataSet()
    trainloader = data.DataLoader(dst, batch_size=4)
    for i in trainloader:
        a,b,c,d = i
        print(b.shape)
