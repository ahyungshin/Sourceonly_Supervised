import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils import data
from model.deeplab_multi import DeeplabMulti
from dataset.gta5_dataset import GTA5DataSet
# from dataset.synthia_dataset import SYNTHIADataSet
from dataset.crosscity_dataset import CrossCityDataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.vistas_dataset import MapillaryDataSet


# NEED TO CHANGE ############
DATA_DIRECTORY_TARGET = '/data/dataset/CityScapes/'
DATA_LIST_TARGET = 'dataset/cityscapes_list/test.txt'
NUM_CLASSES = 19
SET = 'val'
#############################
# DATA_DIRECTORY_TARGET = '/data/dataset/IDD'
# DATA_LIST_TARGET = './dataset/idd_list/val.txt'
# NUM_CLASSES = 19
# SET = 'val'
# #############################
# DATA_DIRECTORY_TARGET = '/data/dataset/MapillaryVistas'
# DATA_LIST_TARGET = './dataset/mapillary_list/val.txt'
# NUM_CLASSES = 19
# SET = 'validation'
#############################

PER_CLASS = True
SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000
BATCH_SIZE = 1
IGNORE_LABEL = 255

SET_TARGET = 'test'


def get_arguments():
	"""Parse all the arguments provided from the CLI.

	Returns:
	  A list of parsed arguments.
	"""
	parser = argparse.ArgumentParser(description="ETM framework")
	parser.add_argument("--mIoUs-per-class", action='store_true', default=PER_CLASS)
	parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET)
	parser.add_argument("--data-list-target", type=str, default=DATA_LIST_TARGET)
	parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
						help="The index of the label to ignore during the training.")
	parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
							 help="Number of images sent to the network in one step.")
	parser.add_argument("--set", type=str, default=SET,
						help="choose evaluation set.")
	parser.add_argument("--set_target", type=str, default=SET_TARGET,
						help="choose evaluation set.")
	parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
						help="Save summaries and checkpoint every often.")
	parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
						help="Number of training steps for early stopping.")
	return parser.parse_args()


def validate(model, target=''):
	args = get_arguments()

	input_size = (1024, 512)

	if args.num_classes == 19:
		name_classes = np.asarray(["road",
								"sidewalk",
								"building",
								"wall",
								"fence",
								"pole",
								"light",
								"sign",
								"vegetation",
								"terrain",
								"sky",
								"person",
								"rider",
								"car",
								"truck",
								"bus",
								"train",
								"motocycle",
								"bicycle"])
	elif args.num_classes == 18:
		name_classes = np.asarray(["road",
								"sidewalk",
								"building",
								"wall",
								"fence",
								"pole",
								"light",
								"sign",
								"vegetation",
								# "terrain",
								"sky",
								"person",
								"rider",
								"car",
								"truck",
								"bus",
								"train",
								"motocycle",
								"bicycle"])



	# Create the model and start the evaluation process
	for files in range(int(args.num_steps_stop / args.save_pred_every)):
		print('Step: ', (files + 1) * args.save_pred_every)
		device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

		interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

		model.eval()
		
		if target == 'cityscapes':
			loader = torch.utils.data.DataLoader(
				cityscapesDataSet(args.data_dir_target, args.data_list_target,
							crop_size=input_size, ignore_label=args.ignore_label, set=args.set),
				batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'idd':
			loader = torch.utils.data.DataLoader(
				IDDDataSet(args.data_dir_target, args.data_list_target, 
							crop_size=(1024, 512), set=args.set, num_classes=18),
							batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'mapillary':
			loader = torch.utils.data.DataLoader(
				MapillaryDataSet(args.data_dir_target, args.data_list_target, 
							crop_size=(1024, 512), set=args.set, num_classes=19),
							batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'rio':
			loader = torch.utils.data.DataLoader(
				CrossCityDataSet(args.data_dir_target, 'Rio',
								 crop_size=input_size, ignore_label=args.ignore_label,
								 set=args.set_target, num_classes=args.num_classes),
				batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'rome':
			loader = torch.utils.data.DataLoader(
				CrossCityDataSet(args.data_dir_target, 'Rome',
								 crop_size=input_size, ignore_label=args.ignore_label,
								 set=args.set_target, num_classes=args.num_classes),
				batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'taipei':
			loader = torch.utils.data.DataLoader(
				CrossCityDataSet(args.data_dir_target, 'Taipei',
								 crop_size=input_size, ignore_label=args.ignore_label,
								 set=args.set_target, num_classes=args.num_classes),
				batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		if target == 'tokyo':
			loader = torch.utils.data.DataLoader(
				CrossCityDataSet(args.data_dir_target, 'Tokyo',
								 crop_size=input_size, ignore_label=args.ignore_label,
								 set=args.set_target, num_classes=args.num_classes),
				batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

		hist = np.zeros((args.num_classes, args.num_classes))
		for i, data in enumerate(loader):
			images_val, labels, _, _ = data
			images_val, labels = images_val.to(device), labels.to(device)
			_, pred = model(images_val) #, input_size)
			pred = interp(pred)#.cpu().data[0].numpy()
			_, pred = pred.max(dim=1)
			
			labels = labels.cpu().numpy()
			pred = pred.cpu().detach().numpy()
			hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
		mIoUs = per_class_iu(hist)
		if args.mIoUs_per_class:
			for ind_class in range(args.num_classes):
				print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
		print('===> mIoU ({}): '.format(target) + str(round(np.nanmean(mIoUs) * 100, 2)))
		print('=' * 50)
		return np.nanmean(mIoUs)


def fast_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
	return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
	  
if __name__ == '__main__':
	main()
