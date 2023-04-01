import torch.nn as nn
import torch.nn.functional as F
import torch

from KD.common import BaseClass


class TemperatureScaledKLDivLoss(nn.Module):
    """
    Temperature scaled Kullback-Leibler divergence loss for knowledge distillation (Hinton et al.,
    https://arxiv.org/abs/1503.02531)

    :param float temperature: parameter for softening the distribution to be compared.
    """

    def __init__(self, temperature):
        super(TemperatureScaledKLDivLoss, self).__init__()
        self.temperature = temperature
        self.kullback_leibler_divergence = nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_pred, y):
        """
        Output the temperature scaled Kullback-Leibler divergence loss for given the prediction and the target.
        :param torch.Tensor y_pred: unnormalized prediction for logarithm of the target.
        :param torch.Tensor y: probabilities representing the target.
        """
        log_p = torch.log_softmax(y_pred / self.temperature, dim=1)
        q = torch.softmax(y / self.temperature, dim=1)

        # Note that the Kullback-Leibler divergence is re-scaled by the squared temperature parameter.
        loss = (self.temperature ** 2) * self.kullback_leibler_divergence(log_p, q)
        return loss


class GaussianLoss(nn.Module):
    """
    Gaussian loss for transfer learning with variational information distillation.
    """

    def forward(self, y_pred, y):
        """
        Output the Gaussian loss given the prediction and the target.
        :param tuple(torch.Tensor, torch.Tensor) y_pred: predicted mean and variance for the Gaussian
        distribution.
        :param torch.Tensor y: target for the Gaussian distribution.
        """
        y_pred_mean, y_pred_var = y_pred
        loss = torch.mean(0.5 * ((y_pred_mean - y) ** 2 / y_pred_var + torch.log(y_pred_var)))
        return loss


class VanillaKD(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf
    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
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
        loss_fn=nn.MSELoss(),
        temp=2.0,
        distil_weight=0.3,
        device="cuda",
        log=False,
        logdir="./Experiments",
        args=None
    ):
        super(VanillaKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            teacher_lrScheduler,
            student_lrScheduler,
            logname,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
            args
        )
        self.loss_fn = TemperatureScaledKLDivLoss(temperature=1.0)  # t越大越平滑
#        self.loss_fn_representation = TemperatureScaledKLDivLoss(temperature=2.0)
        self.vid_loss = GaussianLoss()

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, gt_loss, contrastive_loss):
        """
        Function used for calculating the KD loss during distillation
        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student_loss
        :param contrastive_loss
        """
        gt_loss = 0.25 * gt_loss
#        print('gt loss:', gt_loss)
        
        kl_loss = 0
        for i in range(6):
            out_teacher = y_pred_teacher[i]
            out_student = y_pred_student[i]
            # out_student = out_student[:out_student.size(0)//3]  # only remain student outputs before augmentation
            loss = self.loss_fn(out_student, out_teacher)  # 前者是input，后者是target
            kl_loss += loss   # due to augmentation*3

        kl_loss = 0.75 * kl_loss
#        print('kl loss:', kl_loss)

#        contrastive_loss = 1. * contrastive_loss
#        print('contrastive loss:', contrastive_loss)
        
#        all_loss = kl_loss + gt_loss + contrastive_loss
#        all_loss = contrastive_loss
        all_loss = kl_loss + gt_loss
        
        return all_loss


    def calculate_kd_loss_new(self, y_pred_student, y_pred_teacher, student_features, teacher_features, gt_loss):
    
        gt_loss = 0.25 * gt_loss
#        print('gt loss:', gt_loss)
        
        kl_loss = 0
        for i in range(6):
            out_teacher = y_pred_teacher[i]
            out_student = y_pred_student[i]
            # out_student = out_student[:out_student.size(0)//3]  # only remain student outputs before augmentation
            loss = self.loss_fn(out_student, out_teacher)  # 前者是input，后者是target
            kl_loss += loss   # due to augmentation*3

        kl_loss = 0.75 * kl_loss

        representation_loss = 0.75 * self.loss_fn(student_features, teacher_features)

        all_loss = kl_loss + gt_loss + representation_loss
#        all_loss = kl_loss + gt_loss
#        all_loss = gt_loss + representation_loss
        
        return all_loss


    def calculate_vid_loss(self, y_pred_student, y_pred_teacher, student_loss, student_features, teacher_features):
        """
        CrossEntropyLoss + KD Loss + VID Loss
        :param y_pred_student: Prediction made by the student model
        :param y_pred_teacher: Prediction made by the teacher model
        :param student_loss
        :param student_features: Intermediate embedding in the student model
        :param teacher_features: Intermediate embedding in the teacher model
        :return: loss
        """
        ce_weight = 0.6
        kl_weight = 0.2
        vid_weight = 0.2

        ce_loss = ce_weight * student_loss

        vid_loss = vid_weight * self.vid_loss(student_features, teacher_features)

        kl_loss = 0
        for i in range(6):
            out_teacher = y_pred_teacher[i]
            out_student = y_pred_student[i]
            loss = self.loss_fn(out_student, out_teacher)  # 前者是input，后者是target
            kl_loss += loss / 6
        kl_loss *= kl_weight

        total_loss = ce_loss + kl_loss + vid_loss

        return total_loss

