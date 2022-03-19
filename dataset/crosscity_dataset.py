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

class CrossCityDataSet(data.Dataset):
    def __init__(self, root, city, max_iters=None, crop_size=(1024, 512), ignore_label=255, set='train', num_classes=13):
        self.root = root
        self.city = city
        self.list_path = "./dataset/{}_list/{}.txt".format(city.lower(), set)
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6,
        #                       24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: ignore_label, 13: ignore_label,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: ignore_label,
                              18: ignore_label, 19: 3, 20: 4, 21: 5, 22: ignore_label, 23: 6, 24: 7, 25: 8, 26: 9, 27: ignore_label,
                              28: 10, 29: ignore_label, 30: ignore_label, 31: ignore_label, 32: 11, 33: 12}

        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        for name in self.img_ids:
            img_file = osp.join(self.root, self.city, "Images", self.set.capitalize(), name)
            if self.set == "train":
                self.files.append({
                    "img": img_file,
                    "label": "",
                    "name": name
                })
            else:
                label_file = osp.join(self.root, self.city, "Labels", self.set.capitalize(), name[:-4] + "_eval.png")
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })


    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy


    def __getitem__(self, index):
        datafiles = self.files[index]


        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        
        
        if self.set == "train":
            label = datafiles["label"]
            label_copy = label
        else:
            datafiles = self.files[index]
            label = datafiles["label"]
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (1024,512), Image.NEAREST)
            label = np.asarray(label, np.float32)
            label = self.id2trainId(label)
            size = image.shape

        return image.copy(), label.copy(), np.array(size), name





        # else:
        #     label = Image.open(datafiles["label"])
        #     label = label.resize(self.crop_size, Image.NEAREST)
        #     label = np.asarray(label, np.float32)
        #     label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #     for k, v in self.id_to_trainid.items():
        #         label_copy[label == k] = v

        # return image, label_copy, name


if __name__ == '__main__':
    dst = CrossCityDataSet('/work/NTHU_Datasets', 'Rio',
                      crop_size=(1024, 512), ignore_label=255, set='train', num_classes=13)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, name = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
