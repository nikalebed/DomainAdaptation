import math
import random
import torch


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

