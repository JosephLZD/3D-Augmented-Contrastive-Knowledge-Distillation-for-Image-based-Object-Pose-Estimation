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
	help='index of models, 1: teacher; 2: student_baseline')
parser.add_argument('-d', '--data', type=int, default=3,
	help='index of datasets, 1: Pascal3D; 2: Pix3D;  3: ObjectNet3D')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if args.model == 1:
	net_feat = PoseEstimator(shape='PointCloud', shape_feature_dim=1024,
                              img_feature_dim=1024,
                              azi_classes=24, ele_classes=12, inp_classes=24)
	net_feat.cuda()
	net_feat_student = BaselineEstimator(img_feature_dim=2048, azi_classes=24, ele_classes=12, inp_classes=24)
	net_feat_student.cuda()

if args.model == 1:
	load_checkpoint(net_feat, '/home/d3010/lzd/PoseFromShape/PoseFromShape-master/result/PointCloud_ObjectNet3D/deformNet_new/model_best.pth')
	load_checkpoint(net_feat_student, '/home/d3010/lzd/PoseFromShape/PoseFromShape-master/save_models/ObjectNet3D/best_student_new2.pt')

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
	test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar',
                     'iron',
                     'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub',
                     'wheelchair']
#	test_cats = ['filing_cabinet']

#dataset_val = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
#                                     cat_choice=test_cats)
dataset_val = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                 cat_choice=test_cats,
                                 shape='PointCloud', shape_dir='pointcloud')

data_loader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)

print(len(dataset_val))
##-------------------------------



##-------------------------------
net_feat.eval()
sample_feats = []
sample_labels = []

with torch.no_grad():
    for i, data in enumerate(data_loader):
        im, shapes, label = data
        im = im.cuda()
        shapes = shapes.cuda()
        x, feat_3d_fused, feat_contrastive = net_feat(im, shapes)
#        feat_3d_fused = F.normalize(feat_3d_fused, dim=-1)  # 归一化!
        sample_feats.extend(feat_3d_fused.cpu().numpy())
        sample_labels.extend([0 for i in range(len(label))])
#        feat_contrastive = F.normalize(feat_contrastive, dim=-1)  # 归一化!
        sample_feats.extend(feat_contrastive.cpu().numpy())
        sample_labels.extend([1 for i in range(len(label))])

#        _, feat_student, _ = net_feat_student(im)
#        feat_student = F.normalize(feat_student, dim=-1)  # 归一化!
#        sample_feats.extend(feat_student.cpu().numpy())
#        sample_labels.extend([2 for i in range(len(label))])

sample_feats = np.array(sample_feats)
feats = sample_feats.reshape(-1, sample_feats.shape[-1])

sample_labels = np.array(sample_labels)
labels = sample_labels.reshape(len(sample_labels), 1)

print(feats.shape, labels.shape)
##-------------------------------



##-------------------------------
X = feats
# y = labels[:, 0] // args.bin
y = labels

feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

#X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))


df_subset = df.copy()
data_subset = df_subset[feat_cols].values
##-------------------------------


##-------------------------------
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['tsne-2d-three'] = tsne_results[:,2]
fig = plt.figure(figsize=(16, 10))
ax = Axes3D(fig)
ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c=plt.cm.Set1(y))

#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x="tsne-2d-one", y="tsne-2d-two",
#    hue="y",
#    palette=sns.color_palette("hls", 2),
#    data=df,
#    legend="full",
#    alpha=0.3
#)
##-------------------------------



##-------------------------------
#pca_50 = PCA(n_components=50)
#pca_result_50 = pca_50.fit_transform(data_subset)
#print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

#time_start = time.time()
#tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
##tsne_pca_results = tsne.fit_transform(pca_result_50)
#tsne_pca_results = tsne.fit_transform(data_subset)
#print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
#df_subset['tsne-pca50-one'] = tsne_pca_results[:, 0]
#df_subset['tsne-pca50-two'] = tsne_pca_results[:, 1]
#df_subset['tsne-pca50-three'] = tsne_pca_results[:, 2]
#plt.figure(figsize=(16, 10))
#sns.scatterplot(
#    x="tsne-pca50-one", y="tsne-pca50-two", z="tsne-pca50-three",
#    hue="y",  # label
#    # palette=sns.color_palette("hls", int(360 / args.bin)),
#    palette=sns.color_palette("hls", 2),
#    data=df_subset,
#    legend="full",
#    alpha=0.3
#)


##-------------------------------
model_names = {1: 'MyModel', 2: 'StudentBaseline'}
data_names = {1: 'Pascal3D', 2: 'Pix3D', 3: 'ObjectNet3D'}

fig_path = os.path.join('/home/d3010/lzd/PoseFromShape/PoseFromShape-master', 'vis_feat', 
	'tSNE_{}_{}.png'.format(data_names[args.data], model_names[args.model]))
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path)


