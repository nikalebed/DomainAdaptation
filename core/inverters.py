from utils.II2S import II2S
import torch
from argparse import Namespace
from utils.e4e import e4e
from utils.image_utils import resize_batch


class BaseInverter:
    def __init__(self):
        self.device = 'cuda'

    @torch.no_grad()
    def get_latents(self, imgs_t):
        raise NotImplementedError()


class II2SInverter(BaseInverter):
    def __init__(self):
        super().__init__()
        from utils.II2S_options import II2S_s_opts
        self.ii2s = II2S(II2S_s_opts)

    @torch.no_grad()
    def get_latents(self, imgs_t):
        image_full_res = imgs_t.to(
            self.device)
        image_resized = resize_batch(imgs_t, 256).to(
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
    def get_latents(self, imgs_t):
        imgs_t = resize_batch(imgs_t, 256).to(self.device)
        images, w_plus = self.net(imgs_t, randomize_noise=False, return_latents=True)
        return w_plus
