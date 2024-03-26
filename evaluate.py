import os
import click
from utils.sg2_utils import Inferencer
import torch

DEFAULT_LATENT_DIR = 'example_latents'

def evaluate_model(model, latents):
    metrics = {}


def visualize_model(model, latents):
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
@click.option('--latents_dir', default=DEFAULT_LATENT_DIR, help='directory with latents to calculate metric on')
@click.option('--n_images', default=7, help='number of images to process')
@click.argument('ckpts', help='checkpoint to evaluate', default=None, nargs=-1)
def main(latents_dir, n_images, metrics, ckpt):
    # latents_dir = latents_dir if latents_dir else DEFAULT_LATENT_DIR
    latents = get_latents(latents_dir)
    net = Inferencer(ckpt)
    src, trg = net(latents)
    print()


if __name__ == '__main__':
    main()
