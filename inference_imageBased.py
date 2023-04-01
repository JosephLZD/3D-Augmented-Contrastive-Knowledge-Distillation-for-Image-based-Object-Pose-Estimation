import argparse
import os, sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from auxiliary.model import BaselineEstimator
from auxiliary.dataset import normalize
from auxiliary.dataset import resize_pad
from auxiliary.utils import load_checkpoint

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# basic settings
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None, help='resume pth')
parser.add_argument('--gpu', type=str, default="0", help='gpu index')

# model hyper-parameters
parser.add_argument('--img_size', type=int, default=224, help='input image size')
parser.add_argument('--img_feature_dim', type=int, default=2048, help='feature dimension for images')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

opt = parser.parse_args()
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

model = BaselineEstimator(img_feature_dim=opt.img_feature_dim,
                          azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)

model.cuda()
if not os.path.isfile(opt.ckpt):
    raise ValueError('Non existing file: {0}'.format(opt.ckpt))
else:
    load_checkpoint(model, opt.ckpt)

model.eval()
# ========================================================== #

print('Input image: {} \n ----------------------'.format(opt.img_path))

im_transform = transforms.Compose([transforms.ToTensor(), normalize])
im = Image.open(opt.img_path).convert('RGB')
im_copy = im.copy()
im = resize_pad(im, opt.img_size)
im = im_transform(im)
im = im.unsqueeze(0)
im = im.cuda()

with torch.no_grad():
    out, _ = model(im)
    vp_pred = model.compute_vp_pred(out)

# predictions for original and flipped images
vp_pred = vp_pred.cpu().numpy().squeeze()
vp_pred[1] -= 90
vp_pred[2] -= 180
print('viewpoint prediction:\n Azimuth={}\n Elevation={}\n Inplane Rotation={}'.format(
    vp_pred[0], vp_pred[1], vp_pred[2]))