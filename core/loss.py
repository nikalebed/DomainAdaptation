import torch.nn as nn
from core import lpips
import typing as tp
import torch
import torch.nn.functional as F
from facial_recognition.model_irse import Backbone


def cosine_loss(x, y):
    return 1.0 - F.cosine_similarity(x, y)


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


def direction_loss(
        clip_batch
) -> torch.Tensor:
    trg_encoded, src_encoded = clip_batch['trg_encoded'], clip_batch['src_encoded']
    trg_domain_emb, src_domain_emb = clip_batch['trg_domain_emb'], clip_batch['src_domain_emb']

    edit_im_direction = trg_encoded - src_encoded
    edit_domain_direction = trg_domain_emb - src_domain_emb

    if trg_domain_emb.ndim == 3:
        edit_domain_direction = edit_domain_direction.mean(axis=1)

    return cosine_loss(edit_im_direction, edit_domain_direction).mean()


# attentive style-loss
def clip_difa_local(
        clip_batch: tp.Dict[str, tp.Dict]
):
    tgt_tokens, src_tokens = clip_batch['trg_tokens'], clip_batch['src_tokens']
    B, N, _ = tgt_tokens.shape
    style_tokens = clip_batch['trg_tokens_style'].repeat(B, 1, 1)

    tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)
    style_tokens /= style_tokens.clone().norm(dim=-1, keepdim=True)

    attn_weights = torch.bmm(tgt_tokens, style_tokens.permute(0, 2, 1))

    cost_matrix = 1 - attn_weights

    row_values, row_indices = cost_matrix.min(dim=2)
    col_values, col_indices = cost_matrix.min(dim=1)

    row_sum = row_values.mean(dim=1)
    col_sum = col_values.mean(dim=1)

    overall = torch.stack([row_sum, col_sum], dim=1)
    return overall.max(dim=1)[0].mean()


# SCC
class PSPLoss(torch.nn.Module):
    def __init__(self, device='cuda', num_keep_first=7):
        super(PSPLoss, self).__init__()

        self.device = device
        self.num_keep_first = num_keep_first
        self.psp_loss_type = 'dynamic'
        self.delta_w_type = 'mean'
        self.sliding_window_size = 50
        # self.weight = weight
        self.psp_alpha = 0.6
        self.source_set = []
        self.target_set = []
        self.source_pos = 0
        self.target_pos = 0
        self.iter = 0

    def update_queue(self, src_vec, tgt_vec):
        if len(self.target_set) < self.sliding_window_size:
            self.source_set.append(src_vec.mean(0).detach())
            self.target_set.append(tgt_vec.mean(0).detach())
        else:
            self.source_set[self.source_pos] = src_vec.mean(0).detach()
            self.source_pos = (self.source_pos + 1) % self.sliding_window_size
            self.target_set[self.target_pos] = tgt_vec.mean(0).detach()
            self.target_pos = (self.target_pos + 1) % self.sliding_window_size

    def multi_stage_loss(self, target_encodings, source_encodings):
        if self.cond is not None:
            target_encodings = self.cond * target_encodings
            source_encodings = self.cond * source_encodings
        return F.l1_loss(target_encodings, source_encodings)

    def constrained_loss(self, cond):
        return torch.abs(cond.mean(1) - self.psp_alpha).mean()

    def update_w(self, source_encodings, target_encodings):
        if self.delta_w_type == 'mean':
            self.update_queue(source_encodings, target_encodings)
            self.source_mean = torch.stack(self.source_set).mean(0, keepdim=True)
            self.target_mean = torch.stack(self.target_set).mean(0, keepdim=True)
            # Get the editing direction
            delta_w = self.target_mean - self.source_mean
        return delta_w

    def dynamic_loss(self, target_encodings, source_encodings, delta_w):
        # Get the conditional vector to mask special enough channels
        delta_w = delta_w.flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        cond = torch.zeros(num_channel).to(self.device)
        cond[chosen_order] = 1
        cond = cond.unsqueeze(0)

        # Get masked encodings
        target_encodings = cond * target_encodings
        source_encodings = cond * source_encodings

        # # Update the mean direction of target domain and difference
        # self.iter_diff.append(torch.abs(cond - self.cond).sum().cpu().item() / len(delta_w))
        # self.iter_mean.append(cond.mean().cpu().item())
        # self.iter_sim.append(self.cosine_similarity(delta_w, self.target_direction).sum().cpu().item())

        loss = F.l1_loss(target_encodings, source_encodings)
        return loss

    def forward(self, batch):
        batch_size = batch['inv_data']['trg_latents'].shape[0]
        target_encodings = batch['inv_data']['trg_latents'].reshape(batch_size, -1)
        source_encodings = batch['inv_data']['src_latents'].reshape(batch_size, -1)

        iters = batch['inv_data']['iters']

        # Mask w+ codes controlling style and fine details
        if self.num_keep_first > 0:
            keep_num = self.num_keep_first * 512
            target_encodings = target_encodings[:, 0:keep_num]
            source_encodings = source_encodings[:, 0:keep_num]

        if self.psp_loss_type == "multi_stage":
            # edit_direction = target_encodings - source_encodings
            # theta = (edit_direction.clone() * self.target_direction).sum(dim=-1, keepdim=True)
            # return F.l1_loss(edit_direction, theta * self.target_direction)
            # loss = self.multi_stage_loss(target_encodings, source_encodings)
            ...
        elif self.psp_loss_type == "dynamic":
            delta_w = self.update_w(source_encodings, target_encodings)
            regular_weight = max(0, \
                                 (iters - self.sliding_window_size) / (
                                         self.iter - self.sliding_window_size))
            loss = regular_weight * self.dynamic_loss(target_encodings, source_encodings, delta_w=delta_w)
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")

        return loss


class IDLoss(nn.Module):
    def __init__(self, model_path):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_path))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        # self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, batch, src=True):
        if src:
            y = batch['pixel_data']['src_img']
        else:
            y = batch['pixel_data']['trg_ref']
        y_hat = batch['pixel_data']['trg_img']

        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        f