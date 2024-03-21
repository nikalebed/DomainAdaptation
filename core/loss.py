import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from core import lpips


class LossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(LossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]
        self.l2 = torch.nn.MSELoss()
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()
        # self.percept = VGGLoss()

    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)

    def _loss_lpips(self, gen_im, ref_im, **kwargs):

        return self.percept(gen_im, ref_im).sum()

    def forward(self, ref_im_H, ref_im_L, gen_im_H, gen_im_L):

        loss = 0
        loss_fun_dict = {
            'l2': self._loss_l2,
            'percep': self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'l2':
                var_dict = {
                    'gen_im': gen_im_H,
                    'ref_im': ref_im_H,
                }
            elif loss_type == 'percep':
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight * tmp_loss
        return loss, losses


def global_loss(batch):
    v_dom = batch[]
    v_samp = batch['img_B_encoded'] - batch['img_A_encoded']
    return 1.0 - F.cosine_similarity(x, y)


def get_loss(name):
    if name == '':
        pass
    elif name == '':
        pass
    elif name == '':
        pass
    else:
        raise ValueError(name)


class ComposedLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()
        self.config = loss_config
        self.loss_funcs = loss_config.loss_funcs

    def forward(self, batch):
        losses = {}
        for loss in self.loss_funcs:
            losses[loss] = get_loss[loss](batch)
        losses['total'] = sum(v for v in losses.values())
        return losses
