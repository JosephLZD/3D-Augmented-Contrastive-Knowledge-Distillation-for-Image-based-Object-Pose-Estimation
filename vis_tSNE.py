import argparse
import os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from auxiliary.model import PoseEstimator, BaselineEstimator

import torch
from torch.utils.data import DataLoader

from auxiliary.dataset import Pascal3D, ShapeNet, Pix3D, Pascal3DContrast
from auxiliary.utils import KaiMingInit, save_checkpoint, load_checkpoint


import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import time

import torch.nn.functional as F



import argparse
parser = argparse.ArgumentParser()

# network training procedure settings
parser.add_argument('-n', '--nb', type=int, default=1)
parser.add_argument('-m', '--model', type=int, default=1,
	help='index of models, 1: mymodel; 2: student_baseline')
parser.add_argument('-d', '--data', type=int, default=3,
	help='index of datasets, 1: Pascal3D; 2: Pix3D;  3: ObjectNet3D')
parser.add_argument('-b', '--bin', type=int, default=35, 
	help='azimuth angle bin size for each cluster')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

net_feat = BaselineEstimator(img_feature_dim=2048, azi_classes=24, ele_classes=12, inp_classes=24)
net_feat.cuda()

if args.model == 1:
	load_checkpoint(net_feat, '/home/d3010/lzd/PoseFromShape/PoseFromShape-master/save_models/ObjectNet3D/best_student_new2.pt')

if args.model == 2:
	load_checkpoint(net_feat, '/home/d3010/lzd/PoseFromShape/PoseFromShape-master/save_models/ObjectNet3D/best_student_new2.pt')



##-------------------------------
if args.data == 1:
	root_dir = os.path.join('/home/d3010/lzd/PoseFromShape/PoseFromShape-master/data/Pascal3D')
	annotation_file = 'Pascal3D.txt'

if args.data == 2:
	root_dir = os.path.join('/home/d3010/lzd/PoseFromShape/PoseFromShape-master/data/Pix3D')
	annotation_file = 'Pix3D.txt'

if args.data == 3:
	root_dir = os.path.join('/home/d3010/lzd/PoseFromShape/PoseFromShape-master/data/ObjectNet3D')
	annotation_file = 'ObjectNet3D.txt'
#	test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar',
#                     'iron',
#                     'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub',
#                     'wheelchair']
	test_cats = ['bed', 'bookshelf', 'calculator', 'computer', 'door', 'filing_cabinet', 'microwave', 'stove', 'toilet']

dataset_val = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                     cat_choice=test_cats)
data_loader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)

print(len(dataset_val))
##-------------------------------



##-------------------------------
net_feat.eval()
sample_feats = []
sample_labels = []

with torch.no_grad():
    for i, data in enumerate(data_loader):
        im, label = data
        im = im.cuda()
        _,  feat = net_feat(im)
#        feat = F.normalize(feat, dim=-1)  # 归一化!
        sample_feats.extend(feat.cpu().numpy())
        sample_labels.extend(label.numpy())

sample_feats = np.array(sample_feats)
feats = sample_feats.reshape(-1, sample_feats.shape[-1])

sample_labels = np.array(sample_labels)
labels = sample_labels.reshape(-1, sample_labels.shape[-1])

print(feats.shape, labels.shape)
##-------------------------------

##-------------------------------
X_old = feats
y = labels[:, 0] // args.bin

# 统计标签对应的最少样本数量
#count_label = {}
#for idx in range(len(labels)):
#    label = labels[idx]
#    if label[0] <= 90:
#        if (label[2] // args.bin) not in count_label:
#            count_label[label[2] // args.bin] = 0
#        count_label[label[2] // args.bin] += 1
#
#nums = []
#for label in count_label:
#    nums.append(count_label[label])
#    count_label[label] = 0  # 准备之后再次计数
#
#constrain_num = np.mean(nums)
#print('约束数量：', constrain_num)

X = []
y = []
# 给标签分类
for idx in range(len(labels)):
   label = labels[idx]
   feat = X_old[idx]
#   if label[0] <= 90:  
#       X.append(feat)
#       if label[1] <= 90:
#           if label[2] <= 120:
#               y.append('0')
#           elif 120 < label[2] <= 240:
#               y.append('1')
#           else:
#               y.append('2')
#       elif 90 < label[1] <= 180:
#           if label[2] <= 120:
#               y.append('3')
#           elif 120 < label[2] <= 240:
#               y.append('4')
#           else:
#               y.append('5')
#   if label[0] <= 120:
#       X.append(feat)
#       y.append(label[2] // args.bin)
#       else:
#           if label[2] <= 120:
#               y.append('6')
#           elif 120 < label[2] <= 240:
#               y.append('7')
#           else:
#               y.append('8')
#      if count_label[label[1] // args.bin] < constrain_num:
#      count_label[label[2] // args.bin] += 1
   X.append(feat)
   y.append(label[1] // args.bin)
       

X = np.array(X)
X = X.reshape(-1, X.shape[-1])
y = np.array(y)
y = y.reshape(-1, 1)

print(X.shape, y.shape)


feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

num_colors = len(np.unique(y))

#X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))


df_subset = df.copy()
data_subset = df_subset[feat_cols].values
##-------------------------------


##-------------------------------
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)
# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1] 
# df['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(df)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", int(360 / args.bin)),
#     data=df,
#     legend="full",
#     alpha=0.3
# )
##-------------------------------



##-------------------------------
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# 对数据进行归一化操作
#x_min, x_max = np.min(tsne_pca_results, 0), np.max(tsne_pca_results, 0)
#tsne_pca_results = tsne_pca_results / (x_max - x_min)

df_subset['tsne-pca50-one'] = tsne_pca_results[:, 0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:, 1]
plt.figure()
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",  # label
#    palette=sns.color_palette("hls", int(360 / args.bin)),
    palette=sns.color_palette("hls", num_colors),
    data=df_subset,
    legend="full",
    alpha=0.3
)

#fig = plt.figure(figsize=(16, 10))
#ax = Axes3D(fig)
#ax.scatter(tsne_pca_results[:,0], tsne_pca_results[:,1], tsne_pca_results[:,2], c=plt.cm.Set1(y))

##-------------------------------
model_names = {1: 'MyModel', 2: 'StudentBaseline'}
data_names = {1: 'Pascal3D', 2: 'Pix3D', 3: 'ObjectNet3D'}

fig_path = os.path.join('/home/d3010/lzd/PoseFromShape/PoseFromShape-master', 'vis_feat', '{}'.format(args.nb), 'bin{}'.format(args.bin),
	'tSNE_{}_{}_bin={}.png'.format(data_names[args.data], model_names[args.model], args.bin))
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path)


