import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
<<<<<<< HEAD
# from tensorboardX import SummaryWriter
=======
>>>>>>> target supervised

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
<<<<<<< HEAD
from dataset.cityscapes_dataset import cityscapesDataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/root/agilab22/3CUDA/GTA5/data'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 13 #19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()
=======

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

SOURCE = 'gta5'
DATA_DIR = '/data/dataset/GTA5'
DATA_LIST = './dataset/gta5_list/train.txt'

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 1
INPUT_SIZE = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth' #'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LAMBDA_SEG = 0.1


def get_arguments():
	"""Parse all the arguments provided from the CLI.

	Returns:
	  A list of parsed arguments.
	"""
	parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
	parser.add_argument("--model", type=str, default=MODEL,
						help="available options : DeepLab")
	parser.add_argument("--data-dir", type=str, default=DATA_DIR,
						help="available options : DeepLab")
	parser.add_argument("--data-list", type=str, default=DATA_LIST,
						help="available options : DeepLab")
	parser.add_argument("--source", type=str, default=SOURCE,
						help="available options : cityscapes")
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
						help="Number of images sent to the network in one step.")
	parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
						help="Accumulate gradients for ITER_SIZE iterations.")
	parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
						help="number of workers for multithread dataloading.")
	parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
						help="Comma-separated string with height and width of source images.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
						help="Base learning rate for training with polynomial decay.")
	parser.add_argument("--momentum", type=float, default=MOMENTUM,
						help="Momentum component of the optimiser.")
	parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
						help="Number of classes to predict (including background).")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
						help="Number of training steps.")
	parser.add_argument("--power", type=float, default=POWER,
						help="Decay parameter to compute the learning rate.")
	parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
						help="Random seed to have reproducible results.")
	parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
						help="Where restore model parameters from.")
	parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
						help="lambda_seg.")
	parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
						help="Where to save snapshots of the model.")
	parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
						help="Regularisation parameter for L2-loss.")
	parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
	return parser.parse_args()
>>>>>>> target supervised


args = get_arguments()


def loss_calc(pred, label, device):
<<<<<<< HEAD
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d().to(device)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # w, h = map(int, args.input_size_target.split(','))
    # input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()

        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    # Cityscapes Dataloader
    # trainloader = data.DataLoader(cityscapesDataSet( max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                                  scale=False), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                pin_memory=True)

    # GTA5 Dataloader
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)


    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    for i_iter in range(args.num_steps):

        loss_seg_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)


        for sub_i in range(args.iter_size):

            _, batch = trainloader_iter.__next__()


            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            _, pred2 = model(images) #[1,19,41,41] 19 classes
            pred2 = interp(pred2)

            # proper normalization
            loss = loss_calc(pred2, labels, device) / args.iter_size
            loss.backward()
            
            loss_seg_value2 += loss.item()

        optimizer.step()

        if i_iter % 1000==0:
            print('exp = {}'.format(args.snapshot_dir))
            print('iter = {0:8d}/{1:8d}, loss_seg2 = {2:.3f}'.format(i_iter, args.num_steps, loss_seg_value2))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
   #         break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
=======
	"""
	This function returns cross entropy loss for semantic segmentation
	"""
	# out shape batch_size x channels x h x w -> batch_size x channels x h x w
	# label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
	label = Variable(label.long()).to(device)
	criterion = CrossEntropy2d().to(device)

	return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
	lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1:
		optimizer.param_groups[1]['lr'] = lr * 10

def main():
	"""Create the model and start the training."""

	device = torch.device("cuda" if not args.cpu else "cpu")

	w, h = map(int, args.input_size.split(','))
	input_size = (w, h)

	cudnn.enabled = True

	# Create network
	if args.model == 'DeepLab':
		model = DeeplabMulti(num_classes=args.num_classes)
		if args.restore_from[:4] == 'http' :
			saved_state_dict = model_zoo.load_url(args.restore_from)
		else:
			saved_state_dict = torch.load(args.restore_from)

		new_params = model.state_dict().copy()

		for i in saved_state_dict:
			# Scale.layer5.conv2d_list.3.weight
			i_parts = i.split('.')
			# if not args.num_classes == 19 or not i_parts[1] == 'layer5':
			if not i_parts[1] == 'layer5':
				new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
		model.load_state_dict(new_params)


	model.train()
	model.to(device)

	cudnn.benchmark = True

	if not os.path.exists(args.snapshot_dir):
		os.makedirs(args.snapshot_dir)

	# GTA5 Dataloader
	if args.source == 'gta5':
		trainloader = data.DataLoader(
			GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
						crop_size=input_size, mean=IMG_MEAN),
			batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	trainloader_iter = enumerate(trainloader)


	optimizer = optim.SGD(model.optim_parameters(args),
						  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	optimizer.zero_grad()

	interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
	current_loss = 0
	best_loss = 1e9

	for i_iter in range(args.num_steps):

		loss_seg_value1 = 0
		loss_seg_value2 = 0

		optimizer.zero_grad()
		adjust_learning_rate(optimizer, i_iter)

		for sub_i in range(args.iter_size):

			_, batch = trainloader_iter.__next__()
			images, labels, _, _ = batch
			images = Variable(images).to(device)
			labels = labels.long().to(device)

			pred1, pred2 = model(images)
			pred1 = interp(pred1)
			pred2 = interp(pred2)

			loss_seg1 = loss_calc(pred1, labels, device)
			loss_seg2 = loss_calc(pred2, labels, device)
			loss = loss_seg2 + args.lambda_seg * loss_seg1

			# proper normalization
			loss = loss / args.iter_size
			loss.backward()
			loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
			loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

		optimizer.step()

		if i_iter > 0 and i_iter%100 == 0:
			print('exp = {}'.format(osp.join(args.snapshot_dir)))
			print('iter = {0:8d}/{1:8d}'.format(i_iter, args.num_steps))
			print('loss_seg1 = {0:.3f} loss_seg2 = {1:.3f}'.format(loss_seg_value1,  loss_seg_value2))

			# Snapshots directory
			if not os.path.exists(osp.join(args.snapshot_dir)):
				os.makedirs(osp.join(args.snapshot_dir))

			current_loss = loss_seg_value1 + loss_seg_value2
			if current_loss < best_loss:
				best_loss = current_loss
				print('Succesfully save best model ... loss: ')
				print(best_loss)
				torch.save(model.state_dict(),
						osp.join(args.snapshot_dir, 'best.pth'))

if __name__ == '__main__':
	main()
>>>>>>> target supervised
