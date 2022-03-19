import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
import json
import cv2

class MapillaryDataSet(data.Dataset):
	def __init__(self, root, img_list_path, max_samples=1000, crop_size=(1280,720), transform=None, set='train', num_classes=19, ignore_label=255):
		self.root = root
		self.crop_size = crop_size

		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
		self.files = []
		self.set = set
		for img_name in self.img_ids:
			img_file = osp.join(self.root, "%s/images/%s" % (set, img_name))
			# img_name = img_name # jpg -> png
			label_file = osp.join(self.root, "%s/v1.2/labels/%s" % (set, img_name[:-3]+'png'))
			self.files.append({
				"img": img_file,
				"label": label_file,
				"name": img_name
			})
		self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
		# self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
		# 					  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
		# 					  7: 0, 8: 0, 9: ignore_label, 10: ignore_label, 11: 1, 12: 1, 13: 1,
		# 					  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 2,
		# 					  18: ignore_label, 19: 2, 20: 2, 21: 3, 22: 3, 23: 4, 24: 4, 25: 5, 26: 6, 27: 6,
		# 					  28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 6}

		self.id_to_trainid = {13:0, 24:0, 41:0, 2:1, 15:1, 17:2, 6:3, 3:4, 45:5, 47:5,
							48:6, 50:7, 30:8, 29:9, 27:10, 19:11, 20:12, 21:12, 22:12,
							55:13, 61:14, 54:15, 58:16, 57:17, 52:18}

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
		image = image.resize(self.crop_size, Image.BICUBIC)
		image = np.asarray(image, np.float32)
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1))
		size = image.shape

		name = datafiles["name"]

		label = datafiles["label"]
		label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
		label = cv2.resize(label, (1024,512), Image.NEAREST)
		label = np.asarray(label, np.float32)

		label_copy = 255 * np.ones(label.shape, dtype=np.float32)
		for k, v in self.id_to_trainid.items():
				label_copy[label == k] = v

		return image.copy(), label_copy, np.array(size), name


if __name__ == '__main__':
	dst = GTA5DataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()
