import os
import click
from utils.sg2_utils import Inferencer
import torch
from omegaconf import OmegaConf
from utils.image_utils import get_image_t, construct_image_grid
from utils.common import get_style_latent
from PIL import Image
import numpy as np
from core.inverters import get_inverter

DEFAULT_CONFIG_DIR = 'configs'
DEFAULT_IMAGE_DIR = 'images'


def get_latents(dir, n_examples=None):
    latents = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames)[:n_examples]:
            path = os.path.join(root, fname)
            latents.append(torch.load(path))
    return torch.cat(latents)


def visualize_style_mixing(ckpt_name):
    zs = []
    for seed in range(1, 8):
        zs += [torch.from_numpy(np.random.RandomState(seed).randn(1, 512))]
    zs = [torch.cat(zs).to(torch.float32).cuda()]
    inverter = get_inverter('e4e')

    model = Inferencer(ckpt_name)
    img_t = get_image_t(model.config.training.target_class)
    latents = inverter.get_latents(img_t).squeeze()

    src, trg = model(zs, input_is_latent=False)
    res = [src, trg]
    for alpha in [0.1, 0.3, 0.5, 0.7, 1]:
        _, trg = model(zs, style_mixing={
            'alpha': alpha,
            'm': 7,
            'ref_latents': latents}, input_is_latent=False)
        res += [trg]
    arr = construct_image_grid(res, size=256)
    return Image.fromarray(arr.astype('uint8'), 'RGB')


@click.command()
@click.argument('config_name', default='eval.yaml')
def main(config_name):
    config_path = os.path.join(DEFAULT_CONFIG_DIR, config_name)
    config = OmegaConf.load(config_path)
    latents = get_latents(config.latents_dir, config.n_examples)
    latents = latents.to('cuda')

    styles = []
    src = None
    rows = []
    for ckpt in config.ckpts:
        net = Inferencer((os.path.join(config.checkpoints_dir, ckpt)))
        if src is None:
            src, _ = net([latents], input_is_latent=True)

        style_path = net.config.training.target_class
        styles.append(get_image_t(style_path))

        style_latents = torch.zeros((18, 512))
        if config.style_mixing.alpha > 0:
            style_latents = get_style_latent(config.style_mixing.inversion_type, styles[-1], style_path).squeeze(0)
        style_latents = style_latents.cuda()

        _, trg = net([latents], style_mixing={
            'alpha': config.style_mixing.alpha,
            'm': 7,
            'ref_latents': style_latents}, input_is_latent=True)

        rows.append(trg)

    arr = construct_image_grid(header=src, index=styles, imgs_t=rows, size=config.img_size)
    img = Image.fromarray(arr.astype('uint8'), 'RGB')
    img.save(os.path.join(DEFAULT_IMAGE_DIR, config.res_name))


if __name__ == '__main__':
    main()
