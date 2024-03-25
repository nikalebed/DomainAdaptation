import math
import random
import torch
from torch import nn
from core.da_model import DomainAdaptationGenerator
from core.parametrization import Parametrization


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


def get_trainable_model_state(config, state_dict):
    if config.training.patch_key == "original":
        # Save TunningGenerator as state_dict
        ckpt = {
            "model_type": 'original',
            "state_dict": state_dict
        }
    else:
        # save parametrization
        ckpt = {
            "model_type": "parameterization",
            "patch_key": config.training.patch_key,
            "state_dict": state_dict
        }

    ckpt['sg2_params'] = dict(config.generator_args['stylegan2'])
    return ckpt


class Inferencer(nn.Module):
    def __init__(self, ckpt, device):
        super().__init__()
        ckpt = get_trainable_model_state(ckpt['config'], ckpt['trainable'])
        self.device = device
        self.source_generator = DomainAdaptationGenerator(
            **self.ckpt['sg2_params'])
        # self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.add_patches()  # TODO add options
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

        self.trainable = Parametrization(
            get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']))

        self.model_da.load_state_dict(ckpt['state_dict'])
        self.model_da.to(self.device).eval()

    @torch.no_grad()
    def forward(self, latents, **kwargs):
        if not kwargs.get('input_is_latent', False):
            latents = self.source_generator.style(latents)
            kwargs['input_is_latent'] = True

        src_imgs, _ = self.source_generator(latents, **kwargs)
        if not kwargs.get('truncation', False):
            kwargs['truncation'] = 1

        # if self.da_type == 'im2im':
        #     latents = self._mtg_mixing_noise(latents, truncation=kwargs['truncation'])#TODO???
        #     kwargs.pop('truncation')

        if self.model_type == 'original':
            trg_imgs, _ = self.model_da(latents, **kwargs)
        else:
            trg_imgs, _ = self.sg2_source(latents, params=self.model_da(), **kwargs)

        return src_imgs, trg_imgs
