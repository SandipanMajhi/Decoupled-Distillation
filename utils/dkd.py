import torch
import torch.nn as nn 
import torch.nn.functional as F


class DKD(nn.Module):
    def __init__(self, student, teacher, ce_loss_weight = 0.5, temperature = 1.0, alpha = 0.5, beta = 0.5):
        super().__init__()

        self.student = student
        self.teacher = teacher
        self.ce_loss_weight = ce_loss_weight

        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def get_gt_mask(self, logits, targets):
        targets = targets.reshape(-1)
        mask = torch.zeros_like(logits)
        mask[range(logits.shape[0]), targets] = 1
        mask = mask.bool()
        return mask
    
    def get_other_mask(self, logits, targets):
        targets = targets.reshape(-1)
        mask = torch.ones_like(logits)
        mask[range(logits.shape[0]), targets] = 0
        mask = mask.bool()
        return mask
    
    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim = 1, keepdims = True)
        t2 = (t * mask2).sum(dim = 1, keepdims = True)
        return torch.cat([t1, t2], dim = 1)


    def dkd_loss(self, logits_student, logits_teacher, targets, alpha, beta, temperature):
        gt_mask = self.get_gt_mask(logits_student, targets)
        other_mask = self.get_other_mask(logits_student, targets)

        pred_student = F.softmax(logits_student / temperature, dim = 1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim = 1)

        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)

        log_pred_student = torch.log(pred_student)
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, size_average=False) * (temperature**2) / targets.shape[0]

        log_pred_student2 = F.log_softmax(logits_student/temperature - 1000 * gt_mask, dim = 1)
        pred_teacher2 = F.softmax(logits_teacher/temperature - 1000 * gt_mask, dim = 1)

        nckd_loss = F.kl_div(log_pred_student2, pred_teacher2, size_average=False) * (temperature**2) / targets.shape[0]

        return alpha * tckd_loss + beta * nckd_loss


    def forward_train(self, x, target):
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        if self.training:
            loss_ce = self.ce_loss_weight * F.cross_entropy(student_logits, target)
            loss_dkd = self.dkd_loss(student_logits, teacher_logits, target, self.alpha, self.beta, self.temperature)
            return loss_ce + loss_dkd, student_logits.detach()
        else:
            loss_ce = self.ce_loss_weight * F.cross_entropy(student_logits, target)
            return loss_ce, student_logits.detach()
    

    def forward(self, x, target):
        return self.forward_train(x, target)

