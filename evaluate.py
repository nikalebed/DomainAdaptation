import os
import click
from utils.sg2_utils import Inferencer
import torch
from omegaconf import OmegaConf
from utils.image_utils import get_image_t, construct_image_grid

DEFAULT_CONFIG_DIR = 'configs'


def evaluate_model(model, latents):
    metrics = {}


def visualize_model(model, styles, latents):
    ...


def get_latents(dir):
    latents = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            latents.append(torch.load(path))
    return latents


@click.command()
@click.argument('config_name', default='eval.yaml')
def main(config_name):
    config_path = os.path.join(DEFAULT_CONFIG_DIR, config_name)
    config = OmegaConf.load(config_path)
    latents = get_latents(config.latents_dir)[:config.n_examples]

    styles = []
    src = None
    rows = []

    for ckpt in config.ckpts:
        net = Inferencer(ckpt)
        styles.append(get_image_t(net.config.target_class))
        src, trg = net(latents)
        rows.append(trg)
    return construct_image_grid(header=src, index=styles, imgs_t=rows, size=256)


if __name__ == '__main__':
    main()
