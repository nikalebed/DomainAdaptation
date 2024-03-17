import torch
from core.da_model import DomainAdaptationGenerator
from core.parametrization import Parametrization

from utils.sg2_utils import get_stylegan_conv_dimensions, mixing_noise
from core.loss import ComposedLoss
from utils.common import load_clip
from core.dataset import ImagesDataset
import typing as tp
from utils.II2S import II2S


class DomainAdaptationTrainer:
    def setup_clip_encoders(self):
        self.clip_encoders = {}
        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.clip_encoders[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

    def clip_encode_image(self, model, image, preprocess):
        image_features = model.encode_image(preprocess(image))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def setup_inverter(self):
        self.ii2s = II2S(self.config)
        pass

    def invert_images_ii2s(self, image_info):
        image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(
            self.device)
        image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(
            self.device)

        latents, = self.ii2s.invert_image(
            image_full_res,
            image_resized
        )

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.source_generator = None
        self.trainable = None
        self.criterion = None
        self.optimizer = None
        self.bicubic = None
        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_latent = None
        self.style_image_inverted_A = None
        self.clip_encoders = None

    def setup(self):
        self.setup_source_generator()
        self.setup_trainable()
        self.setup_criterion()

    def setup_source_generator(self):
        self.source_generator = DomainAdaptationGenerator(
            **self.config.generator_args)
        self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

    def setup_trainable(self):
        self.trainable = Parametrization(
            get_stylegan_conv_dimensions(self.source_generator.generator.size))

    def setup_criterion(self):
        self.criterion = ComposedLoss(self.config.loss_config)

    def setup_style_image(self):
        style_image_info = ImagesDataset(self.config.img_opt, self.config.style_image)[0]

        self.style_image_full_res = style_image_info[
            'image_high_res_torch'].unsqueeze(0).to(self.device)
        self.style_image_resized = style_image_info[
            'image_low_res_torch'].unsqueeze(
            0).to(self.device)

        self.style_image_latent = self.invert_image(
            style_image_info).detach().clone()

        self.style_image_inverted_A = self.forward_source(
            [self.style_image_latent], input_is_latent=True)

    @torch.no_grad()
    def forward_source(self, latents, **kwargs) -> torch.Tensor:
        sampled_images, _ = self.source_generator(latents, **kwargs)
        return sampled_images.detach()

    def forward_trainable(self, latents, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if self.config.training.patch_key == "original":
            sampled_images, _ = self.trainable(
                latents, **kwargs
            )
            offsets = None
        else:
            offsets = self.trainable()
            sampled_images, _ = self.source_generator(
                latents, offsets=offsets, **kwargs
            )

        return sampled_images, offsets

    def encode_batch(self, sample_z):
        img_A = self.forward_source(sample_z)
        img_B, offsets = self.forward_trainable(sample_z)
        style_image_inverted_B, _ = self.forward_trainable(
            [self.style_image_latent], input_is_latent=True
        )
        batch = {}
        for visual_encoder_key, (
                model, preprocess) in self.clip_encoders.items():
            img_A_encoded = self.clip_encode_image(model, img_A,
                                                   preprocess)
            img_B_encoded = self.clip_encode_image(model, img_B, preprocess)

            style_img_encoded = self.clip_encode_image(model,
                                                       self.style_image_full_res,
                                                       preprocess)
            style_img_inverted_A_encoded = self.clip_encode_image(model,
                                                                  self.style_image_inverted_A,
                                                                  preprocess)
            style_img_inverted_B_encoded = self.clip_encode_image(model,
                                                                  style_image_inverted_B,
                                                                  preprocess)
            batch[visual_encoder_key] = {'img_A_encoded': img_A_encoded,
                                         'img_B_encoded': img_B_encoded,
                                         'style_img_encoded': style_img_encoded,
                                         'style_img_inverted_A_encoded': style_img_inverted_A_encoded,
                                         'style_img_inverted_B_encoded': style_img_inverted_B_encoded}
        return batch

    def train_iter(self):
        z = mixing_noise(self.config.batch, 512, self.config.mixing,
                         self.config.device)
        batch = self.encode_batch(z)
        loss = self.criterion(batch)
        self.optimizer.zero_grad()
        loss['final'].backward()
        self.optimizer.step()
        return loss

    def train(self):
        for i in range(self.config.train_iters):
            loss = self.train_iter()
            if i % self.config.log_iters():
                self.log()

        # generator.load_state_dict(ckpt["g"])
        # discriminator.load_state_dict(ckpt["d"])
        # g_ema.load_state_dict(ckpt["g_ema"])
        #
        # g_optim.load_state_dict(ckpt["g_optim"])
        # d_optim.load_state_dict(ckpt["d_optim"])
