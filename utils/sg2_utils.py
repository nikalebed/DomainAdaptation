import math
import random
import torch
from torch import nn
from core.da_model import DomainAdaptationGenerator
from core.parametrization import Parametrization
from utils.image_utils import get_image_t
from utils.common import get_style_latent


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


def get_stylegan_conv_dimensions(size, channel_multiplier=2):
    channels = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
    }

    log_size = int(math.log(size, 2))
    conv_dimensions = [(channels[4], channels[4])]

    in_channel = channels[4]

    for i in range(3, log_size + 1):
        out_channel = channels[2 ** i]

        conv_dimensions.append((in_channel, out_channel))
        conv_dimensions.append((out_channel, out_channel))

        in_channel = out_channel

    return conv_dimensions


class Inferencer(nn.Module):
    def __init__(self, ckpt_dir, device='cuda'):
        super().__init__()
        ckpt = torch.load(ckpt_dir, map_location=device)
        self.config = ckpt['config']
        self.device = device
        self.source_generator = DomainAdaptationGenerator(
            **self.config.generator_args[self.config.training.generator], device=device)
        self.source_generator.add_patches()  # TODO add options
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

        self.style_path = self.config.training.target_class

        if self.config.training.da_type == 'original':
            self.model_da = DomainAdaptationGenerator(
                **self.config.generator_args[self.config.training.generator])
            self.model_da.add_patches()
        else:
            self.model_da = Parametrization(
                get_stylegan_conv_dimensions(self.config.generator_args.stylegan2.img_size))
        self.model_da = self.model_da.to(self.device)
        self.model_da.load_state_dict(ckpt['trainable'])
        self.model_da.eval()

        self.style_mixing = None
        self.style_latents = None

    def add_style_mixing(self, mixing_config):
        self.style_mixing = mixing_config
        style_img = get_image_t(self.style_path)
        self.style_latents = get_style_latent(mixing_config.inversion_type, style_img, self.style_path).squeeze(0)

    @torch.no_grad()
    def forward(self, latents, **kwargs):
        if not kwargs.get('input_is_latent', False):
            latents = self.source_generator.style(latents)
            kwargs['input_is_latent'] = True

        src_imgs, _ = self.source_generator(latents, **kwargs)

        if self.style_mixing is not None:
            if latents[0].ndim < 3:
                latents[0] = latents[0].unsqueeze(1).repeat(1, 18, 1)
            latents[0][:, 7:] = (1 - self.style_mixing.alpha) * latents[0][:, 7:] + \
                                self.style_mixing.alpha * self.style_latents[7:]

        if not kwargs.get('truncation', False):
            kwargs['truncation'] = 1
        if self.config.training.da_type == 'original':
            trg_imgs, _ = self.model_da(latents, **kwargs)
        else:
            trg_imgs, _ = self.source_generator(latents, params=self.model_da(), **kwargs)

        return src_imgs.detach().cpu(), trg_imgs.detach().cpu()
