import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy
import os

from auxiliary.utils import KaiMingInit, save_checkpoint, save_checkpoint_raw, load_checkpoint, AverageValueMeter, \
    get_pred_from_cls_output, get_pred_from_cls_output_contrast, plot_loss_fig, plot_acc_fig, rotation_acc
from auxiliary.loss import CELoss, DeltaLoss
from tqdm import tqdm
from auxiliary.model_utils import poseNCE, infoNCE_KD, poseNCE_KD, multiposeNCE_KD, singleinfoNCE_KD
from auxiliary.dataset import Pascal3DContrast
from evaluation import test_category_training
import numpy as np

class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework
    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
            self,
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            teacher_lrScheduler=None,
            student_lrScheduler=None,
            logname=None,
            loss_fn=nn.KLDivLoss(),
            temp=2.0,
            distil_weight=0.5,
            device="cuda",
            log=False,
            logdir="./Experiments",
            args=None
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student

        if teacher_lrScheduler:
            self.teacher_lrScheduler = teacher_lrScheduler
        if student_lrScheduler:
            self.student_lrScheduler = student_lrScheduler

        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir
        self.logname = logname
        self.device = device
        self.args = args
        print('device:', self.device)

        #        if self.log:
        #            self.writer = SummaryWriter(logdir)

        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)

        self.criterion_azi = CELoss(360).to(self.device)
        self.criterion_ele = CELoss(180).to(self.device)
        self.criterion_inp = CELoss(360).to(self.device)
        self.bin_size = 15  # bin_size=15
        self.criterion_reg = DeltaLoss(self.bin_size).to(self.device)

    def train_teacher(
            self,
            epochs=20,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/teacher.pt",
    ):
        """
        Function that will be training the teacher
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        teacher_train_acc_rot = AverageValueMeter()
        self.teacher_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Teacher... ")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            # update learning rate
            self.teacher_lrScheduler.step()
            for i, data in enumerate(self.train_loader):
                im, shapes, label = data
                im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
                out = self.teacher_model(im, shapes)

                # compute losses and update the meters
                loss_azi = self.criterion_azi(out[0], label[:, 0])
                loss_ele = self.criterion_ele(out[1], label[:, 1])
                loss_inp = self.criterion_inp(out[2], label[:, 2])
                loss_reg = self.criterion_reg(out[3], out[4], out[5], label.float())
                loss = loss_azi + loss_ele + loss_inp + loss_reg

                # compute rotation matrix accuracy
                preds = get_pred_from_cls_output([out[0], out[1], out[2]])
                for n in range(len(preds)):
                    pred_delta = out[n + 3]
                    delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
                    preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                acc_rot = rotation_acc(
                    torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                    label.float())
                teacher_train_acc_rot.update(acc_rot.item(), im.size(0))

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss += loss

            epoch_acc = teacher_train_acc_rot.avg

            epoch_val_acc = self.evaluate(teacher=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            #            if self.log:
            #                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
            #                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)
            #                self.writer.add_scalar(
            #                    "Validation accuracy/Teacher", epoch_val_acc, epochs
            #                )

            # save losses and accuracies into log file
            with open(self.logname, 'a') as f:
                text = str('Teacher Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_acc %.2f \n \n' %
                           (epochs, epoch_loss, epoch_acc, epoch_val_acc))
                f.write(text)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )

            self.post_epoch_call(ep)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def _train_student(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/student.pt",
    ):
        """
        Function to train student model - for internal use only.
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        student_train_acc_rot = AverageValueMeter()
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            # update learning rate
            self.student_lrScheduler.step()

            for i, data in enumerate(self.train_loader):
                im, shapes, label = data
                im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
                out = self.student_model(im)
                teacher_out = self.teacher_model(im, shapes)

                # compute losses and update the meters
                loss_azi = self.criterion_azi(out[0], label[:, 0])
                loss_ele = self.criterion_ele(out[1], label[:, 1])
                loss_inp = self.criterion_inp(out[2], label[:, 2])
                loss_reg = self.criterion_reg(out[3], out[4], out[5], label.float())
                student_loss = loss_azi + loss_ele + loss_inp + loss_reg

                # kd loss
                loss = self.calculate_kd_loss(out, teacher_out, student_loss)

                # compute rotation matrix accuracy
                preds = get_pred_from_cls_output([out[0], out[1], out[2]])
                for n in range(len(preds)):
                    pred_delta = out[n + 3]
                    delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
                    preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                acc_rot = rotation_acc(
                    torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                    label.float())
                student_train_acc_rot.update(acc_rot.item(), im.size(0))

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            epoch_acc = student_train_acc_rot.avg

            _, epoch_val_acc = self._evaluate_model(self.student_model, teacher=False, verbose=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            #            if self.log:
            #                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
            #                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
            #                self.writer.add_scalar(
            #                    "Validation accuracy/Student", epoch_val_acc, epochs
            #                )
            with open(self.logname, 'a') as f:
                text = str('Student Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_acc %.2f \n \n' %
                           (ep + 1, epoch_loss, epoch_acc, epoch_val_acc))
                f.write(text)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def _train_student_crd(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./save_models/best_student.pt"
    ):
        """
        Function to train student model - for internal use only.
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        best_loss = 100000.
        # self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in range(epochs):
            # reset to train mode because we set them eval in evaluate function!
            self.student_model.train()
            epoch_loss = 0.0
            correct = 0
            student_train_acc_rot = AverageValueMeter()
            student_train_loss = AverageValueMeter()

            for i, data in enumerate(self.train_loader):
                # im, shapes, label = data
                im, shapes, label, im_flip, label_flip, im_rot, label_rot = data  # without im_pos
                im, shapes, label, im_flip, label_flip, im_rot, label_rot = \
                    im.cuda(), shapes.cuda(), label.cuda(), im_flip.cuda(), \
                    label_flip.cuda(), im_rot.cuda(), label_rot.cuda()

                # b means batch_size
                b = im.shape[0]

                # concatenate flipped images
                im = torch.cat((im, im_flip), 0)
                label = torch.cat((label, label_flip), 0)

                # concatenate rotated images
                im = torch.cat((im, im_rot), 0)
                label = torch.cat((label, label_rot), 0)

                # forward pass
#                _, feat = self.student_net_feat(im)  # resnet做embedding
                out, student_features = self.student_model(im)  # mlp做rotation预测 net_vp

                # need to get 2 more batch_size of shapes for the data augmentation setting
                shapes = torch.cat((shapes, shapes, shapes), 0)  # TODO: whether change shapes according to imgs aug?
                teacher_out, _, teacher_features = self.teacher_model(im, shapes)

                # compute losses and update the meters
                loss_azi = self.criterion_azi(out[0], label[:, 0])
                loss_ele = self.criterion_ele(out[1], label[:, 1])
                loss_inp = self.criterion_inp(out[2], label[:, 2])
                loss_reg = self.criterion_reg(out[3], out[4], out[5], label.float())
                student_loss = loss_azi + loss_ele + loss_inp + loss_reg  # CE Loss
#                student_loss = 0.

                # contrastive loss between student and teacher
#                loss_poseNCE_s2t_1 = infoNCE_KD(student_features[:b, :], teacher_features[:b, :],
#                                       label[:b, :], self.args.tau)
#                loss_poseNCE_s2t_2 = infoNCE_KD(student_features[b:2*b, :], teacher_features[b:2*b, :],
#                                       label[b:2*b, :], self.args.tau)
#                loss_poseNCE_s2t_3 = infoNCE_KD(student_features[2*b:3*b, :], teacher_features[2*b:3*b, :],
#                                       label[2*b:3*b, :], self.args.tau)
#                loss_poseNCE = 0.3 * loss_poseNCE_s2t_1 + 0.3 * loss_poseNCE_s2t_2 + 0.3 * loss_poseNCE_s2t_3
#                loss_poseNCE = 0.


                # kd loss
#                loss = self.calculate_kd_loss(out, teacher_out, student_loss, loss_poseNCE)
#                loss = student_loss + loss_poseNCE
                loss = self.calculate_kd_loss_new(out, teacher_out, student_features, teacher_features, student_loss)

                # compute rotation matrix accuracy
                preds = get_pred_from_cls_output([out[0], out[1], out[2]])
                for n in range(len(preds)):
                    pred_delta = out[n + 3]
                    bs = pred_delta.size(0)
                    delta_value = pred_delta[torch.arange(bs), preds[n].long()].tanh() / 2
                    preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                acc_rot = rotation_acc(
                    torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                    label.float())

                student_train_loss.update(loss.item(), im.size(0))
                student_train_acc_rot.update(acc_rot.item(), im.size(0))

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            # update learning rate
            self.student_lrScheduler.step()
            
            epoch_acc = student_train_acc_rot.avg
            epoch_loss = student_train_loss.avg

            epoch_val_acc, epoch_val_med = self._evaluate_model(self.student_model, teacher=False, verbose=True)
#            epoch_val_loss = self._evaluate_model_self_supervised(self.student_model, verbose=True)

            save_checkpoint_raw({
                'epoch': ep,
                'student_model': self.student_model.state_dict()
            }, './save_models/checkpoint.pth')

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                if save_model:
                    torch.save(self.student_model.state_dict(), save_model_pth)

#            if epoch_val_loss < best_loss:
#                best_loss = epoch_val_loss
#                if save_model:
#                    torch.save(self.student_model.state_dict(), "./save_models/best_student_self_supervised.pt")
#
            with open(self.logname, 'a') as f:
                text = str('Student Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_acc %.2f -- val_med %.2f  \n \n' %
                           (ep + 1, epoch_loss, epoch_acc, epoch_val_acc, epoch_val_med))
                f.write(text)
#            with open(self.logname, 'a') as f:
#                text = str('Student Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_loss %.2f  \n \n' %
#                           (ep + 1, epoch_loss, epoch_acc, epoch_val_loss))
#                f.write(text)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )
        if plot_losses:
            plt.plot(loss_arr)

    def _train_stage_1(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_teacher_model_pth="./save_models/s1_best_teacher.pt",
            save_student_model_pth="./save_models/s1_best_student.pt"
    ):
        """
        Function to train student model - for internal use only.
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        """
        self.teacher_model.train()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        best_loss = 100000.
        # self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_teacher_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Stage 1...")

        for ep in range(epochs):
            # reset to train mode because we set them eval in evaluate function!
            self.teacher_model.train()
            self.student_model.train()
            epoch_loss = 0.0
            train_acc_rot = AverageValueMeter()
            train_loss = AverageValueMeter()

            for i, data in enumerate(self.train_loader):
                # im, shapes, label = data
                im, shapes, label = data
                im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()

                _, student_features = self.student_model(im)  # mlp做rotation预测 net_vp

                teacher_out, teacher_features = self.teacher_model(im, shapes)

                # compute losses and update the meters of the teacher
                loss_azi = self.criterion_azi(teacher_out[0], label[:, 0])
                loss_ele = self.criterion_ele(teacher_out[1], label[:, 1])
                loss_inp = self.criterion_inp(teacher_out[2], label[:, 2])
                loss_reg = self.criterion_reg(teacher_out[3], teacher_out[4], teacher_out[5], label.float())
                teacher_loss = loss_azi + loss_ele + loss_inp + loss_reg  # CE Loss

                # contrastive loss between student and teacher
                loss_poseNCE_s2t = infoNCE_KD(student_features, teacher_features,
                                          label, self.args.tau)
                loss_poseNCE_t2s = infoNCE_KD(teacher_features, student_features,
                                          label, self.args.tau)
                loss_poseNCE = 0.5 * loss_poseNCE_s2t + 0.5 * loss_poseNCE_t2s

                loss = teacher_loss + 0.75 * loss_poseNCE

                # compute rotation matrix accuracy
                preds = get_pred_from_cls_output([teacher_out[0], teacher_out[1], teacher_out[2]])
                for n in range(len(preds)):
                    pred_delta = teacher_out[n + 3]
                    bs = pred_delta.size(0)
                    delta_value = pred_delta[torch.arange(bs), preds[n].long()].tanh() / 2
                    preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                acc_rot = rotation_acc(
                    torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                    label.float())

                train_loss.update(loss.item(), im.size(0))
                train_acc_rot.update(acc_rot.item(), im.size(0))

                self.optimizer_student.zero_grad()
                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_student.step()
                self.optimizer_teacher.step()

                epoch_loss += loss.item()

            # update learning rate
            self.student_lrScheduler.step()
            self.teacher_lrScheduler.step()

            epoch_acc = train_acc_rot.avg
            epoch_loss = train_loss.avg

            epoch_val_acc, epoch_val_med = self._evaluate_model(self.student_model, teacher=False, verbose=True)
            epoch_val_loss = self._evaluate_model_self_supervised(self.student_model, verbose=True)

            save_checkpoint_raw({
                'epoch': ep,
                'teacher_model': self.teacher_model.state_dict(),
                'student_model': self.student_model.state_dict()
            }, './save_models/checkpoint.pth')

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                if save_model:
                    torch.save(self.teacher_model.state_dict(), save_teacher_model_pth)
                    torch.save(self.student_model.state_dict(), save_student_model_pth)

            with open(self.logname, 'a') as f:
                text = str(
                    'Student Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_acc %.2f -- val_med %.2f -- val_contrastive_loss %.2f  \n \n' %
                    (ep + 1, epoch_loss, epoch_acc, epoch_val_acc, epoch_val_med, epoch_val_loss))
                f.write(text)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )
        if plot_losses:
            plt.plot(loss_arr)

    def _train_stage_2(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_student_model_pth="./save_models/s2_best_student.pt"
    ):
        """
        Function to train student model - for internal use only.
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        best_loss = 100000.
        # self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_student_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Stage 2...")

        for ep in range(epochs):
            # reset to train mode because we set them eval in evaluate function!
            self.student_model.train()
            epoch_loss = 0.0
            train_acc_rot = AverageValueMeter()
            train_loss = AverageValueMeter()

            for i, data in enumerate(self.train_loader):
                im, shapes, label, im_flip, label_flip, im_rot, label_rot = data  # without im_pos
                im, shapes, label, im_flip, label_flip, im_rot, label_rot = \
                    im.cuda(), shapes.cuda(), label.cuda(), im_flip.cuda(), \
                    label_flip.cuda(), im_rot.cuda(), label_rot.cuda()

                # b means batch_size
                b = im.shape[0]

                # concatenate flipped images
                im = torch.cat((im, im_flip), 0)
                label = torch.cat((label, label_flip), 0)

                # concatenate rotated images
                im = torch.cat((im, im_rot), 0)
                label = torch.cat((label, label_rot), 0)

                out, _ = self.student_model(im)  # mlp做rotation预测 net_vp

                # need to get 2 more batch_size of shapes for the data augmentation setting
                shapes = torch.cat((shapes, shapes, shapes), 0)  
                teacher_out, _ = self.teacher_model(im, shapes)

                # compute losses and update the meters of the teacher
                loss_azi = self.criterion_azi(out[0], label[:, 0])
                loss_ele = self.criterion_ele(out[1], label[:, 1])
                loss_inp = self.criterion_inp(out[2], label[:, 2])
                loss_reg = self.criterion_reg(out[3], out[4], out[5], label.float())
                student_loss = loss_azi + loss_ele + loss_inp + loss_reg  # CE Loss

                loss = self.calculate_kd_loss(out, teacher_out, student_loss, 0.)

                # compute rotation matrix accuracy
                preds = get_pred_from_cls_output([out[0], out[1], out[2]])
                for n in range(len(preds)):
                    pred_delta = out[n + 3]
                    bs = pred_delta.size(0)
                    delta_value = pred_delta[torch.arange(bs), preds[n].long()].tanh() / 2
                    preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                acc_rot = rotation_acc(
                    torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                    label.float())

                train_loss.update(loss.item(), im.size(0))
                train_acc_rot.update(acc_rot.item(), im.size(0))

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            # update learning rate
            self.student_lrScheduler.step()

            epoch_acc = train_acc_rot.avg
            epoch_loss = train_loss.avg

            epoch_val_acc, epoch_val_med = self._evaluate_model(self.student_model, teacher=False, verbose=True)

            save_checkpoint_raw({
                'epoch': ep,
                'student_model': self.student_model.state_dict()
            }, './save_models/checkpoint.pth')

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                if save_model:
                    torch.save(self.student_model.state_dict(), save_student_model_pth)

            with open(self.logname, 'a') as f:
                text = str(
                    'Student Epoch: %03d || train_loss %.2f || train_acc %.2f -- val_acc %.2f -- val_med %.2f  \n \n' %
                    (ep + 1, epoch_loss, epoch_acc, epoch_val_acc, epoch_val_med))
                f.write(text)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )
        if plot_losses:
            plt.plot(loss_arr)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations
        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, teacher, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.
        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        if self.args.contrast or self.args.crd or self.args.stage == 2:
            Accs = {}
            Meds = {}
            if self.args.dataset == 'ObjectNet3D':
                test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
            elif self.args.dataset == 'Pascal3D':
                test_cats = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
            root_dir = os.path.join('data', self.args.dataset)
            annotation_file = '{}.txt'.format(self.args.dataset)
            for cat in test_cats:
                dataset_test = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                    cat_choice=[cat], keypoint=False, random=False, novel=False,
                                    shape=None)
                Accs[cat], Meds[cat] = test_category_training(None, self.args.batch_size, dataset_test, model, self.args.bin_size, cat)
            mean_Acc = np.array(list(Accs.values())).mean()
            mean_Med = np.array(list(Meds.values())).mean()

            return mean_Acc, mean_Med

        elif self.args.stage == 1:
            Accs = {}
            Meds = {}
            test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar',
                         'iron',
                         'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub',
                         'wheelchair']
            root_dir = os.path.join('data', self.args.dataset)
            annotation_file = '{}.txt'.format(self.args.dataset)
            for cat in test_cats:
                dataset_test = Pascal3DContrast(train=False, root_dir=root_dir, annotation_file=annotation_file,
                                                cat_choice=[cat], keypoint=False, random=False, novel=False,
                                                shape=self.args.shape, shape_dir=self.args.shape_dir)
                Accs[cat], Meds[cat] = test_category_training(self.args.shape, self.args.batch_size, dataset_test, self.teacher_model,
                                                              self.args.bin_size, cat)
            mean_Acc = np.array(list(Accs.values())).mean()
            mean_Med = np.array(list(Meds.values())).mean()

            return mean_Acc, mean_Med

        else:
            val_acc_rot = AverageValueMeter()
            model.eval()
            length_of_dataset = len(self.val_loader.dataset)
            correct = 0
            outputs = []
            predictions = torch.zeros([1, 3], dtype=torch.float).cuda()
            labels = torch.zeros([1, 3], dtype=torch.long).cuda()

            with torch.no_grad():
                for i, data in enumerate(tqdm(self.val_loader)):
                    # load data and label
                    if teacher:
                        im, shapes, label = data
                        im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
                    else:
                        im, shapes, label = data
                        im, label = im.cuda(), label.cuda()

                    # forward pass
                    out = model(im) if teacher is False else model(im, shapes)

                    # compute losses and update the meters
                    if self.criterion_reg is not None:
                        loss_azi = self.criterion_azi(out[0], label[:, 0])
                        loss_ele = self.criterion_ele(out[1], label[:, 1])
                        loss_inp = self.criterion_inp(out[2], label[:, 2])
                        loss_reg = self.criterion_reg(out[3], out[4], out[5], label.float())
                        loss = loss_azi + loss_ele + loss_inp + loss_reg
                        # val_loss.update(loss.item(), im.size(0))

                    # transform the output into the label format
                    preds = get_pred_from_cls_output([out[0], out[1], out[2]])
                    for n in range(len(preds)):
                        pred_delta = out[n + 3]
                        delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
                        preds[n] = (preds[n].float() + delta_value + 0.5) * self.bin_size
                    pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1)

                    # compute accuracy
                    acc_rot = rotation_acc(pred, label.float())
                    val_acc_rot.update(acc_rot.item(), im.size(0))

                    # concatenate results and labels
                    # labels = torch.cat((labels, label), 0)
                    predictions = torch.cat((predictions, pred), 0)

            predictions = predictions[1:, :]
            # labels = labels[1:, :]
            return predictions, val_acc_rot.avg

    def _evaluate_model_self_supervised(self, model, verbose=True):
        model.eval()
        self.teacher_model.eval()
        val_loss = AverageValueMeter()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader)):
                # load data and label
                im, shapes, label = data
                im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()

                # forward pass
                out, student_features = model(im)

                teacher_out, teacher_features = self.teacher_model(im, shapes)

                loss_poseNCE_s2t = infoNCE_KD(student_features, teacher_features,
                                              label, self.args.tau)
                loss_poseNCE_t2s = infoNCE_KD(teacher_features, student_features,
                                              label, self.args.tau)

                loss_poseNCE = 0.75 * (0.5 * loss_poseNCE_s2t + 0.5 * loss_poseNCE_t2s)

                val_loss.update(loss_poseNCE.item(), im.size(0))

        return val_loss.avg

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network
        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy = self._evaluate_model(model, teacher)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.
        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass