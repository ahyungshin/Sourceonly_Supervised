import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
<<<<<<< HEAD

=======
import sys
# np.set_printoptions(threshold=sys.maxsize)
>>>>>>> target supervised

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

<<<<<<< HEAD

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


=======
def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

>>>>>>> target supervised
def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

<<<<<<< HEAD

def compute_mIoU(gt_dir='', pred_dir='', devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info_13class.json'), 'r') as fp:
=======
def compute_mIoU(dataset, gt_dir='', pred_dir='', devkit_dir=''):
    size =(1024,512)
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
>>>>>>> target supervised
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

<<<<<<< HEAD
    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
            
=======
    if dataset == 'idd' or dataset == 'mapillary':
        image_path_list = join(devkit_dir, 'val.txt')
    else:
        image_path_list = join(devkit_dir, 'test.txt')
    

    if dataset == 'cityscapes':
        label_path_list = join(devkit_dir, 'label.txt')
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    elif dataset == 'idd':
        label_path_list = join(devkit_dir, 'val.txt')
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x[:-15]+'gtFine_labelids.png') for x in gt_imgs]
    elif dataset == 'mapillary':
        label_path_list = image_path_list
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x[:-3]+'png' ) for x in gt_imgs]
    else: # cross city
        label_path_list = image_path_list
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x[:-4]+'_eval.png') for x in gt_imgs] # no seperated label.txt

    pred_imgs = open(image_path_list, 'r').read().splitlines()

    if dataset=='mapillary':
        pred_imgs = [join(pred_dir, x.split('/')[-1][:-3] + 'png') for x in pred_imgs]
    else:
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]


    for ind, gt_img in enumerate(gt_imgs):
        pred = np.array(Image.open(pred_imgs[ind]))
        # gt_img = gt_img[:-4]+'_eval'+'.png' #+gt_img[-4:] #cross city
        label = Image.open(gt_imgs[ind])
        label = label.resize(size, Image.NEAREST)
        label = np.asarray(label, np.float32)
        label = label_mapping(label, mapping)

        # pred_c = colorize_mask(pred)
        # pred_c.save('./test_pred/{}.png'.format(ind))
        # label_c = colorize_mask(label)
        # label_c.save('./test_label/{}.png'.format(ind))

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue


>>>>>>> target supervised
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


<<<<<<< HEAD
def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)
=======
# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask


def main(args):
    compute_mIoU(args.dataset, args.gt_dir, args.pred_dir, args.devkit_dir)
>>>>>>> target supervised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--gt_dir', type=str, default='/root/agilab22/3CUDA/Cityscapes/data/gtFine/val', help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', type=str, default='./result/cityscapes', help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
=======

    dataset = 'mapillary' # NEED TO CHANGE


    if dataset == 'cityscapes':
        gt_dir = '/data/dataset/CityScapes/gtFine/val'
        pred_dir = './result/cityscapes'
        devkit_dir = 'dataset/cityscapes_list'
    elif dataset == 'idd':
        gt_dir = '/data/dataset/IDD/gtFine/val'
        pred_dir = './result/idd'
        devkit_dir = 'dataset/idd_list'
    elif dataset == 'mapillary':
        gt_dir = '/data/dataset/MapillaryVistas/validation/v1.2/labels'
        pred_dir = './result/mapillary'
        devkit_dir = 'dataset/mapillary_list'
    elif dataset == 'rio':
        gt_dir = '/data/dataset/NTHU_Datasets/Rio/Labels/Test'
        pred_dir = './result/rio'
        devkit_dir = 'dataset/rio_list'
    elif dataset == 'rome':
        gt_dir = '/data/dataset/NTHU_Datasets/Rome/Labels/Test'
        pred_dir = './result/rome'
        devkit_dir = 'dataset/rome_list'
    elif dataset == 'taipei':
        gt_dir = '/data/dataset/NTHU_Datasets/Taipei/Labels/Test'
        pred_dir = './result/taipei'
        devkit_dir = 'dataset/taipei_list'
    elif dataset == 'tokyo':
        gt_dir = '/data/dataset/NTHU_Datasets/Tokyo/Labels/Test'
        pred_dir = './result/tokyo'
        devkit_dir = 'dataset/tokyo_list'

    parser.add_argument('--dataset', type=str, default=dataset, help='dataset')
    parser.add_argument('--gt_dir', type=str, default=gt_dir, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', type=str, default=pred_dir, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default=devkit_dir, help='base directory of cityscapes')
>>>>>>> target supervised
    args = parser.parse_args()
    print(args.gt_dir)
    main(args)
