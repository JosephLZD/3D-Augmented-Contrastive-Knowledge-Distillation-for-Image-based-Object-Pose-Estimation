import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import os, sys
import time
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator, BaselineEstimator
from auxiliary.dataset import Pascal3D, ShapeNet, Pix3D, Pascal3DContrast
from auxiliary.utils import KaiMingInit, save_checkpoint, load_checkpoint, AverageValueMeter, \
    get_pred_from_cls_output, plot_loss_fig, plot_acc_fig, rotation_acc
from auxiliary.loss import CELoss, DeltaLoss
from evaluation import val
from auxiliary.resnet import resnet50, resnet18

from KD.vision.vanilla import VanillaKD


# =================== DEFINE TRAIN ========================= #
def train(data_loader, model, bin_size, shape, criterion_azi, criterion_ele, criterion_inp, criterion_reg, optimizer):
    train_loss = AverageValueMeter()
    train_acc_rot = AverageValueMeter()

    model.train()

    data_time = AverageValueMeter()
    batch_time = AverageValueMeter()
    end = time.time()
    for i, data in enumerate(data_loader):
        # load data and label
        if shape is not None:
            im, shapes, label = data
            im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
        else:
            im, label = data
            im, label = im.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # forward pass
        out = model(im) if shape is None else model(im, shapes)

        # compute losses and update the meters
        loss_azi = criterion_azi(out[0], label[:, 0])
        loss_ele = criterion_ele(out[1], label[:, 1])
        loss_inp = criterion_inp(out[2], label[:, 2])
        loss_reg = criterion_reg(out[3], out[4], out[5], label.float())
        loss = loss_azi + loss_ele + loss_inp + loss_reg
        train_loss.update(loss.item(), im.size(0))

        # compute rotation matrix accuracy
        preds = get_pred_from_cls_output([out[0], out[1], out[2]])
        for n in range(len(preds)):
            pred_delta = out[n + 3]
            delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
            preds[n] = (preds[n].float() + delta_value + 0.5) * bin_size
        acc_rot = rotation_acc(torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                               label.float())
        train_acc_rot.update(acc_rot.item(), im.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure bacth time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % opt.print_freq == 0:
            print("\tEpoch %3d --- Iter [%d/%d] Train loss: %.2f || Train accuracy: %.2f" %
                  (epoch, i + 1, len(data_loader), train_loss.avg, train_acc_rot.avg))
            print("\tData loading time: %.2f (%.2f)-- Batch time: %.2f (%.2f)\n" %
                  (data_time.val, data_time.avg, batch_time.val, batch_time.avg))

    return [train_loss.avg, train_acc_rot.avg]
# ========================================================== #





if __name__ == '__main__':

    # =================PARAMETERS=============================== #
    parser = argparse.ArgumentParser()

    # network training procedure settings
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
    parser.add_argument('--decrease', type=int, default=100, help='epoch to decrease')
    parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--print_freq', type=int, default=50, help='frequence of output print')

    # model hyper-parameters
    parser.add_argument('--teacher_model', type=str, default=None, help='optional reload teacher model path')
    parser.add_argument('--student_model', type=str, default=None, help='optional reload student model path')
    parser.add_argument('--img_feature_dim', type=int, default=2048, help='feature dimension for images')
    parser.add_argument('--shape_feature_dim', type=int, default=256, help='feature dimension for shapes')
    parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

    # dataset settings
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['ObjectNet3D', 'Pascal3D', 'ShapeNetCore'], help='dataset')
    parser.add_argument('--shape_dir', type=str, default='Renders_semi_sphere',
                        choices=['Renders_semi_sphere', 'pointcloud'], help='subdirectory conatining the shape')
    parser.add_argument('--shape', type=str, default='MultiView',
                        choices=['MultiView', 'PointCloud'], help='shape representation')
    parser.add_argument('--view_num', type=int, default=12, help='number of render images used in each sample')
    parser.add_argument('--tour', type=int, default=2, help='elevation tour for randomized references')
    parser.add_argument('--novel', action='store_true', help='whether to test on novel cats')
    parser.add_argument('--keypoint', action='store_true', help='whether to use only training samples with anchors')
    parser.add_argument('--shot', type=int, default=None, help='K shot number')

    # canonical view randomization as data augmentation
    parser.add_argument('--random', action='store_true', help='activate random canonical view data augmentation')
    parser.add_argument('--random_range', type=int, default=0, help='variation range for randomized references')

    # for KD learning on PoseContrast
    parser.add_argument('--contrast', action='store_true', help='whether do contrastive learning of PoseContrast')
    parser.add_argument('--crd', action='store_true', help='whether do CRD Loss')
    parser.add_argument('--stage', type=int, default=1, help='stage 1 means training the teacher and contrastive learner, and stage 2 means training the student')
    parser.add_argument('--pretrain_backbone', type=str, help='pretrained resnet model path')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--weighting', type=str, default='linear', choices=['linear', 'sqrt', 'square', 'sin', 'sinsin'])

    opt = parser.parse_args()
    print(opt)
    # ========================================================== #

    # ==================RANDOM SEED SETTING===================== #
    opt.manualSeed = 46  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # ========================================================== #

    # =================CREATE DATASET=========================== #
    root_dir = os.path.join('data', opt.dataset)
    annotation_file = '{}.txt'.format(opt.dataset)

    if opt.dataset == 'ObjectNet3D':
        test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar',
                     'iron',
                     'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub',
                     'wheelchair']
        if opt.contrast or opt.crd or opt.stage == 2:
            dataset_train = Pascal3DContrast(train=True, root_dir=root_dir, annotation_file=annotation_file,
                                     cat_choice=test_cats, keypoint=opt.keypoint, novel=opt.novel,
                                     shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                     random_range=opt.random_range, random=opt.random, shot=opt.shot)
            dataset_eval = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                    cat_choice=test_cats, keypoint=opt.keypoint, random=False, novel=opt.novel,
                                    shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour)
        else:  # opt.stage == 1
            dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file,
                                     cat_choice=test_cats, keypoint=opt.keypoint, novel=opt.novel,
                                     shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                     random_range=opt.random_range, random=opt.random)
            dataset_eval = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                    cat_choice=test_cats, keypoint=opt.keypoint, random=False, novel=opt.novel,
                                    shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour)
    elif opt.dataset == 'Pascal3D':
        test_cats = ['bus', 'motorbike'] if opt.novel else None
        if opt.contrast or opt.crd:
            dataset_train = Pascal3DContrast(train=True, root_dir=root_dir, annotation_file=annotation_file,
                                     cat_choice=test_cats, novel=opt.novel,
                                     shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                     random=opt.random, random_range=opt.random_range)
            dataset_eval = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                    shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                    random=False, cat_choice=test_cats, novel=opt.novel)
        else:
            dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file,
                                     cat_choice=test_cats, novel=opt.novel,
                                     shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                     random=opt.random, random_range=opt.random_range)
            dataset_eval = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                    shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                                    random=False, cat_choice=test_cats, novel=opt.novel)
    elif opt.dataset == 'ShapeNetCore':
        # train on synthetic data and evaluate on real data
        bg_dir = os.path.join('data', 'SUN')
        test_root_dir = os.path.join('data', 'Pix3D')
        test_annotation_file = 'Pix3D.txt'
        test_cats = ['2818832', '2871439', '2933112', '3001627', '4256520', '4379243']

        dataset_train = ShapeNet(train=True, root_dir=root_dir, annotation_file=annotation_file, bg_dir=bg_dir,
                                 shape=opt.shape, random=opt.random, cat_choice=test_cats, novel=opt.novel,
                                 view_num=opt.view_num, tour=opt.tour, random_range=opt.random_range)
        dataset_eval = Pix3D(root_dir=test_root_dir, annotation_file=test_annotation_file,
                             shape=opt.shape, view_num=opt.view_num, tour=opt.tour)
    else:
        sys.exit(0)

    train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              drop_last=True)
    eval_loader = DataLoader(dataset_eval, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                             drop_last=True)
    # ========================================================== #

    # ================CREATE NETWORK============================ #
    azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

    student_model = BaselineEstimator(img_feature_dim=2048,
                                  azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
    teacher_model = PoseEstimator(shape=opt.shape, shape_feature_dim=1024,
                              img_feature_dim=1024,
                              azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes,
                              view_num=opt.view_num)

#    # create student feature backbone
#    net_feat = resnet18(num_classes=2048)
#    if opt.pretrain_backbone is not None:
#        print('load pre-trained student resnet base model succeed!')
#        load_checkpoint(net_feat, opt.pretrain_backbone)

    student_model.cuda()
#    net_feat.cuda()
    teacher_model.cuda()

    if opt.teacher_model is not None:
        load_checkpoint(teacher_model, opt.teacher_model)  # 加载teacher的预训练模型
        print('load pre-trained teacher model succeed!')
    else:
        teacher_model.apply(KaiMingInit)
        
    if opt.student_model is not None:
        load_checkpoint(student_model, opt.student_model)  # 加载student的预训练模型
        print('load pre-trained student model succeed!')
    else:
        student_model.apply(KaiMingInit)

    if opt.pretrain_backbone is not None:
        load_checkpoint(student_model.img_encoder, opt.pretrain_backbone)  # 加载student的resnet的预训练模型
        print('load pre-trained student ResNet succeed!')
    # ========================================================== #

    # ================CREATE OPTIMIZER AND LOSS================= #
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=opt.lr, weight_decay=0.0005)
    student_optimizer = optim.Adam(student_model.parameters(), lr=opt.lr, weight_decay=0.0005)
#    net_feat_optimizer = optim.Adam(net_feat.parameters(), lr=opt.lr, weight_decay=0.0005)

    teacher_lrScheduler = optim.lr_scheduler.MultiStepLR(teacher_optimizer, [opt.decrease], gamma=0.1)
    student_lrScheduler = optim.lr_scheduler.MultiStepLR(student_optimizer, [opt.decrease], gamma=0.1)
#    net_feat_lrScheduler = optim.lr_scheduler.MultiStepLR(net_feat_optimizer, [opt.decrease], gamma=0.1)
    # criterion_azi = CELoss(360)
    # criterion_ele = CELoss(180)
    # criterion_inp = CELoss(360)
    # criterion_reg = DeltaLoss(opt.bin_size)
    # ========================================================== #

    # =============DEFINE stuff for logs ======================= #
    # write basic information into the log file
    training_mode = 'baseline_{}'.format(opt.dataset) if opt.shape is None else '{}_{}'.format(opt.shape, opt.dataset)
    if opt.novel:
        training_mode = '{}_novel'.format(training_mode)
    result_path = os.path.join(os.getcwd(), 'result_KD', training_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logname = os.path.join(result_path, 'training_log.txt')
    with open(logname, 'a') as f:
        f.write(str(opt) + '\n' + '\n')
        f.write('training set: ' + str(len(dataset_train)) + '\n')
        f.write('evaluation set: ' + str(len(dataset_eval)) + '\n')

    distiller = VanillaKD(teacher_model, student_model, train_loader, eval_loader,
                          teacher_optimizer, student_optimizer,
                          teacher_lrScheduler=teacher_lrScheduler, student_lrScheduler=student_lrScheduler,
                          logname=logname, device='cuda', args=opt)
    # distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)  # Train the teacher network

    # ================TEST KD================= #
    if opt.contrast:
        distiller._train_student_contrast(epochs=30, plot_losses=True, save_model=True)  # Train the student network
    elif opt.crd:
        distiller._train_student_crd(epochs=60, plot_losses=True, save_model=True)  # Train the student network 
    elif opt.stage == 1:
        distiller._train_stage_1(epochs=300, plot_losses=True, save_model=True)
    elif opt.stage == 2:
        distiller._train_stage_2(epochs=90, plot_losses=True, save_model=True)
        
    # distiller.evaluate(teacher=False)  # Evaluate the student network
    # distiller.get_parameters()  # A utility function to get the number of parameters in the teacher and the student network

