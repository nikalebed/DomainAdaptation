import os
import click
from utils.sg2_utils import Inferencer
import torch
from omegaconf import OmegaConf
from utils.image_utils import get_image_t, construct_image_grid, resize_batch
from PIL import Image
from core import lpips
from torchmetrics.multimodal.clip_score import CLIPScore
from pathlib import Path
from pprint import pprint


class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def calc_metrics(self, src, trg, style):
        src = resize_batch(src, 256).to(self.device)
        trg = resize_batch(trg, 256).to(self.device)
        return {
            'LPIPS': self.lpips(src, trg),
            'CLIPScore': self.clip_score(src, [style] * src.size(0))
        }


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


@click.command()
@click.argument('config_name', default='eval.yaml')
def main(config_name):
    eval = Evaluator()

    config_path = os.path.join(DEFAULT_CONFIG_DIR, config_name)
    config = OmegaConf.load(config_path)
    latents = get_latents(config.latents_dir, config.n_examples)
    latents = latents.to('cuda')

    styles = []
    src = None
    rows = []

    pprint(config)
    metrics = {}

    for ckpt in config.ckpts:
        net = Inferencer((os.path.join(config.checkpoints_dir, ckpt)))
        style_path = net.config.training.target_class

        styles.append(get_image_t(style_path))
        src, trg = net([latents], input_is_latent=True)
        rows.append(trg)

        metrics[net.config.training.target_class] = eval.calc_metrics(src, trg, Path(style_path).stem)

    arr = construct_image_grid(header=src, index=styles, imgs_t=rows, size=config.img_size)
    img = Image.fromarray(arr.astype('uint8'), 'RGB')
    img.save(os.path.join(DEFAULT_IMAGE_DIR, config.res_name))

    pprint(metrics)


if __name__ == '__main__':
    main()
