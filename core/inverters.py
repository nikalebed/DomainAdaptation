from utils.II2S import II2S
import torch
from argparse import Namespace
from utils.e4e import e4e
from utils.image_utils import resize_batch


class BaseInverter:
    def __init__(self):
        self.device = 'cuda'

    def get_latents(self, imgs_t):
        raise NotImplementedError()


def get_inverter(name):
    if name == 'e4e':
        return e4eInverter()
    elif name == 'ii2s':
        return II2SInverter()
    elif name == 'ref':
        return refInverter()
    raise ValueError()


class II2SInverter(BaseInverter):
    def __init__(self):
        super().__init__()
        from utils.II2S_options import II2S_s_opts
        self.ii2s = II2S(II2S_s_opts)

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

    def get_latents(self, imgs_t):
        imgs_t = resize_batch(imgs_t, 256).to(self.device)
        images, w_plus = self.net(imgs_t, randomize_noise=False, return_latents=True)
        return w_plus


from core import lpips
from core.loss import IDLoss
from tqdm import tqdm
from torch.optim import Adam
from gan_models.sg2_model import Generator


class refInverter(BaseInverter):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.id_loss = IDLoss('pretrained/model_ir_se50.pth')
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu='True')
        self.percept.eval()
        self.l2 = torch.nn.MSELoss()

        checkpoint_path = 'pretrained/StyleGAN2/stylegan2-ffhq-config-f.pt'
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.net = Generator(1024, 512, 8,
                             channel_multiplier=2).to(device)
        self.net.load_state_dict(checkpoint["g_ema"], strict=False)
        self.latent_avg = checkpoint["latent_avg"].unsqueeze(0)

    def get_latents(self, imgs_t):
        # latents = self.latent_avg.clone().detach().cuda().unsqueeze(1).repeat(1, 18, 1)
        inverter = e4eInverter()
        latents = inverter.get_latents(imgs_t).clone().detach().cuda()
        latents.requires_grad = True

        opt = Adam([latents], lr=0.01, betas=(0.9, 0.999))

        for i in tqdm(range(500)):
            opt.zero_grad()
            src_inv, _ = self.net([latents], input_is_latent=True)
            loss = self.percept(src_inv, imgs_t).sum()
            loss = loss + self.id_loss({'pixel_data': {
                'src_img': src_inv,
                'trg_img': imgs_t
            }})
            loss = loss + self.l2(src_inv, imgs_t)
            loss.backward()
            opt.step()

        return latents
