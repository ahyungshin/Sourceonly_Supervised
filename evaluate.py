import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from collections import OrderedDict
import os
from PIL import Image
from dataset.crosscity_dataset import CrossCityDataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from dataset.vistas_dataset import MapillaryDataSet

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

EVAL_DATA = 'mapillary' # NEED TO CHANGE


if EVAL_DATA == 'cityscapes':
    DATA_DIRECTORY = '/data/dataset/CityScapes'
    DATA_LIST_PATH = './dataset/cityscapes_list/test.txt'
    SAVE_PATH = './result/cityscapes'
    SET = 'val'
elif EVAL_DATA == 'idd':
    DATA_DIRECTORY = '/data/dataset/IDD'
    DATA_LIST_PATH = './dataset/idd_list/val.txt'
    SAVE_PATH = './result/idd'
    SET = 'val'
elif EVAL_DATA == 'mapillary':
    DATA_DIRECTORY = '/data/dataset/MapillaryVistas'
    DATA_LIST_PATH = './dataset/mapillary_list/val.txt'
    SAVE_PATH = './result/mapillary'
    SET = 'validation'
elif EVAL_DATA == 'rio':
    DATA_DIRECTORY = '/data/dataset/NTHU_Datasets'
    DATA_LIST_PATH = './dataset/rio_list/test.txt'
    SAVE_PATH = './result/rio'
    SET='test'
elif EVAL_DATA == 'rome':
    DATA_DIRECTORY = '/data/dataset/NTHU_Datasets'
    DATA_LIST_PATH = './dataset/rome_list/test.txt'
    SAVE_PATH = './result/rome'
    SET='test'
elif EVAL_DATA == 'taipei':
    DATA_DIRECTORY = '/data/dataset/NTHU_Datasets'
    DATA_LIST_PATH = './dataset/taipei_list/test.txt'
    SAVE_PATH = './result/taipei'
    SET='test'
elif EVAL_DATA == 'tokyo':
    DATA_DIRECTORY = '/data/dataset/NTHU_Datasets'
    DATA_LIST_PATH = './dataset/tokyo_list/test.txt'
    SAVE_PATH = './result/tokyo'
    SET='test'


IGNORE_LABEL = 255
NUM_CLASSES = 19 # NEED TO CHANGE
RESTORE_FROM = './snapshots/best.pth' #'./pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth'
MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--eval_data", type=str, default=EVAL_DATA,
                        help="evaluation dataset.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)
    model.eval()

    if args.eval_data == 'cityscapes':
        testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=1, shuffle=False, pin_memory=True)
    elif args.eval_data == 'idd':
        testloader = data.DataLoader(IDDDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), set=args.set, num_classes=18),
                                        batch_size=1, shuffle=False, pin_memory=True)
    elif args.eval_data == 'mapillary':
        testloader = data.DataLoader(MapillaryDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), set=args.set, num_classes=19),
                                        batch_size=1, shuffle=False, pin_memory=True)
    elif args.eval_data == 'rio':
        testloader = data.DataLoader(CrossCityDataSet(args.data_dir, 'Rio',
                                crop_size=(1024, 512), ignore_label=args.ignore_label,
                                set=args.set, num_classes=args.num_classes),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    elif args.eval_data == 'rome':
        testloader = data.DataLoader(CrossCityDataSet(args.data_dir, 'Rome',
                                crop_size=(1024, 512), ignore_label=args.ignore_label,
                                set=args.set, num_classes=args.num_classes),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    elif args.eval_data == 'taipei':
        testloader = data.DataLoader(CrossCityDataSet(args.data_dir, 'Taipei',
                                crop_size=(1024, 512), ignore_label=args.ignore_label,
                                set=args.set, num_classes=args.num_classes),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    elif args.eval_data == 'tokyo':
        testloader = data.DataLoader(CrossCityDataSet(args.data_dir, 'Tokyo',
                                crop_size=(1024, 512), ignore_label=args.ignore_label,
                                set=args.set, num_classes=args.num_classes),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)




    interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d / %dprocessd' % (index, len(testloader)))
        image, _,_, name = batch
        image = image.to(device)

        if args.model == 'DeeplabMulti':
            output1, output2 = model(image)
            output = interp(output2).cpu().data[0].numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output = model(image)
            output = interp(output).cpu().data[0].numpy()

        output = output.transpose(1,2,0) #[512,1024,13]
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))


if __name__ == '__main__':
    main()