from utils.II2S import II2S
import torch
from argparse import Namespace
from utils.e4e import e4e


class BaseInverter:
    def __init__(self):
        self.device = 'cuda'

    @torch.no_grad()
    def get_latent(self, image_info):
        raise NotImplementedError()


class II2SInverter(BaseInverter):
    def __init__(self):
        super().__init__()
        from utils.II2S_options import II2S_s_opts
        self.ii2s = II2S(II2S_s_opts)

    @torch.no_grad()
    def get_latent(self, image_info):
        if len(image_info['image_high_res_torch'].shape) < 4:
            image_info['image_high_res_torch'] = image_info['image_high_res_torch'].unsqueeze(0)
            image_info['image_low_res_torch'] = image_info['image_low_res_torch'].unsqueeze(0)

        image_full_res = image_info['image_high_res_torch'].to(
            self.device)
        image_resized = image_info['image_low_res_torch'].to(
            self.device)

        latents, = self.ii2s.invert_image(
            image_full_res,
            image_resized
        )
        return latents


class e4eInverter(BaseInverter):
    def __init__(self):
        super().__init__()

        model_path = 'pretrained/e4e_ffhq_encode.pt'
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        self.net = e4e(opts).eval().to(self.device)

    @torch.no_grad()
    def get_latent(self, image_info):
        if len(image_info['image_low_res_torch'].shape) < 4:
            image_info['image_low_res_torch'] = image_info['image_low_res_torch'].unsqueeze(0)

        img = image_info['image_low_res_torch'].to(
            self.device)
        images, w_plus = self.net(img, randomize_noise=False, return_latents=True)
        return w_plus
