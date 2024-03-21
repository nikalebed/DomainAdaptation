import torch
from core.da_model import DomainAdaptationGenerator
from core.parametrization import Parametrization
from utils.sg2_utils import get_stylegan_conv_dimensions, mixing_noise
from core.loss import ComposedLoss
from core.dataset import ImagesDataset
import typing as tp


class DomainAdaptationTrainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.source_generator = None
        self.trainable = None
        self.criterion = None
        self.optimizer = None

        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_latent = None
        self.style_image_inverted_A = None

        self.clip_batch_generator = None

    def setup(self):
        self.setup_source_generator()
        self.setup_trainable()
        self.setup_criterion()
        self.setup_style_image()
        self.setup_clip_batch_generator()

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
        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)

        clip_data = self.clip_batch_generator.calc_batch(frozen_img, trainable_img)

        return {
            'clip_data': clip_data,
            'offsets': offsets,
            'inv_data': inv_data
        }

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
            # if i % self.config.log_iters(): TODO
            #     self.log()
