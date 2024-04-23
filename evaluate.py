import clip
from utils.common import load_clip
import torch.nn.functional as F
import os
from utils.sg2_utils import Inferencer
import torch
from omegaconf import OmegaConf
from core import lpips
from pathlib import Path
from pprint import pprint
from utils.image_utils import resize_batch, get_image_t
from tqdm import tqdm


class Evaluator:
    def __init__(self, device='cuda', n_samples=1000, batch_size=16, seed=0):
        self.device = device
        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.clip_model, self.clip_preprocess = load_clip('ViT-B/16', device)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed

    def encode_image(self, image, norm=True):
        image_features = self.clip_model.encode_image(self.clip_preprocess(image)).detach()
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text, templates=("{}",)):
        text = [t.format(text) for t in templates]
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(tokens).detach().float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def clip_score(self, imgs, text):
        imgs = imgs.to(self.device)
        imgs_encoded = self.encode_image(imgs)
        text_encoded = self.encode_text(text)
        return F.cosine_similarity(imgs_encoded, text_encoded)

    def calc_metrics(self, net):
        style_path = net.config.training.target_class
        style = Path(style_path).stem
        style_img = get_image_t(style_path, size=256)

        torch.manual_seed(self.seed)

        lpips_src = 0
        lpips_ref = 0
        cs = 0

        for i in tqdm(range((self.n_samples + self.batch_size - 1) // self.batch_size)):
            zs = [torch.randn(min(self.batch_size, self.n_samples - i * self.batch_size), 512, device=self.device)]
            src, trg = net(zs)
            lpips_src += self.lpips(resize_batch(src, 256), resize_batch(trg, 256)).sum().item()
            lpips_ref += self.lpips(style_img.repeat(src.size(0), 1, 1, 1), resize_batch(trg, 256)).sum().item()
            cs += self.clip_score(trg, style).sum().item()

        return {
            'LPIPS_src': lpips_src / self.n_samples,
            'LPIPS_ref': lpips_ref / self.n_samples,
            'CLIPScore': cs / self.n_samples
        }


DEFAULT_CONFIG_DIR = 'configs'


def main():
    conf_cli = OmegaConf.from_cli()
    if not conf_cli.get('config_name', False):
        raise ValueError("No config")
    config_path = os.path.join(DEFAULT_CONFIG_DIR, conf_cli.config_name)
    config = OmegaConf.merge(OmegaConf.load(config_path), conf_cli)

    evaluator = Evaluator(n_samples=config.n_samples, batch_size=config.batch_size)

    mean_key = f'mean over {len(config.ckpts)} styles'
    metrics = {mean_key: {
        'LPIPS_src': 0.,
        'LPIPS_ref': 0.,
        'CLIPScore': 0.
    }}

    for ckpt in config.ckpts:
        net = Inferencer((os.path.join(config.checkpoints_dir, ckpt)))
        if config.style_mixing.alpha > 0:
            net.add_style_mixing(config.style_mixing)
        style = Path(net.config.training.target_class).stem
        print(style)
        metrics[style] = evaluator.calc_metrics(net)
        print(metrics[style])
        for key in metrics[mean_key]:
            metrics[mean_key][key] += metrics[style][key]

    for key in metrics[mean_key]:
        metrics[mean_key][key] /= len(config.ckpts)
    pprint(metrics)

    import pandas as pd
    os.makedirs('metrics', exist_ok=True)
    pd.DataFrame.from_dict(metrics, orient='index').to_csv('metrics/' + config.res_name)


if __name__ == '__main__':
    main()
